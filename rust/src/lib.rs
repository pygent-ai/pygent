use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONNECTION, CONTENT_TYPE};
use tokio::sync::{mpsc, Mutex, Notify, Semaphore};
use tokio_util::sync::CancellationToken;

enum StreamItem {
    Data(String),
    Status(u16),
    Error(String),
}

struct ClientState {
    client: reqwest::Client,
    admission: Arc<Semaphore>,
    connections: Arc<Semaphore>,
    admission_limit: usize,
    connection_limit: usize,
    cancel: CancellationToken,
    closed: AtomicBool,
    active: AtomicUsize,
    idle: Notify,
}

struct ActiveGuard(Arc<ClientState>);

impl ActiveGuard {
    fn new(state: Arc<ClientState>) -> Self {
        state.active.fetch_add(1, Ordering::AcqRel);
        Self(state)
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        if self.0.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle.notify_waiters();
        }
    }
}

#[pyclass(module = "pygent._native")]
struct NativeHttpClient {
    state: Arc<ClientState>,
}

#[pymethods]
impl NativeHttpClient {
    #[new]
    #[pyo3(signature = (headers, trust_environment, max_connections, verify_ssl=true))]
    fn new(
        headers: HashMap<String, String>,
        trust_environment: bool,
        max_connections: usize,
        verify_ssl: bool,
    ) -> PyResult<Self> {
        let mut header_map = HeaderMap::with_capacity(headers.len());
        for (name, value) in headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let value = HeaderValue::from_str(&value)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            header_map.insert(name, value);
        }
        let mut builder = reqwest::Client::builder()
            .default_headers(header_map)
            .pool_max_idle_per_host(max_connections)
            .tcp_nodelay(true)
            .danger_accept_invalid_certs(!verify_ssl);
        if !trust_environment {
            builder = builder.no_proxy();
        }
        let client = builder
            .build()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let connection_limit = max_connections.min(32);
        Ok(Self {
            state: Arc::new(ClientState {
                client,
                admission: Arc::new(Semaphore::new(max_connections)),
                connections: Arc::new(Semaphore::new(connection_limit)),
                admission_limit: max_connections,
                connection_limit,
                cancel: CancellationToken::new(),
                closed: AtomicBool::new(false),
                active: AtomicUsize::new(0),
                idle: Notify::new(),
            }),
        })
    }

    #[pyo3(signature = (method, url, body=None, timeout=None))]
    fn request_json<'py>(
        &self,
        py: Python<'py>,
        method: String,
        url: String,
        body: Option<String>,
        timeout: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if state.closed.load(Ordering::Acquire) {
                return Err(PyRuntimeError::new_err("model provider client is closed"));
            }
            let permit = state
                .cancel
                .run_until_cancelled(state.admission.clone().acquire_owned())
                .await
                .ok_or_else(|| PyRuntimeError::new_err("model provider client is closed"))?
                .map_err(|_| PyRuntimeError::new_err("model provider client is closed"))?;
            let _permit = permit;
            let connection = state
                .cancel
                .run_until_cancelled(state.connections.clone().acquire_owned())
                .await
                .ok_or_else(|| PyRuntimeError::new_err("model provider client is closed"))?
                .map_err(|_| PyRuntimeError::new_err("model provider client is closed"))?;
            let _connection = connection;
            let _active = ActiveGuard::new(state.clone());
            let method = reqwest::Method::from_bytes(method.as_bytes())
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let mut request = state
                .client
                .request(method, url)
                .header(CONNECTION, "keep-alive");
            if let Some(body) = body {
                request = request.header(CONTENT_TYPE, "application/json").body(body);
            }
            let operation = async {
                let response = request
                    .send()
                    .await
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                let status = response.status().as_u16();
                let body = response
                    .text()
                    .await
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                Ok((status, body))
            };
            state
                .cancel
                .run_until_cancelled(async {
                    if let Some(seconds) = timeout {
                        tokio::time::timeout(Duration::from_secs_f64(seconds), operation)
                            .await
                            .map_err(|_| PyRuntimeError::new_err("request timed out"))?
                    } else {
                        operation.await
                    }
                })
                .await
                .unwrap_or_else(|| Err(PyRuntimeError::new_err("model provider client is closed")))
        })
    }

    fn stream_sse(&self, url: String, body: String) -> PyResult<NativeSseStream> {
        if self.state.closed.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("model provider client is closed"));
        }
        let (sender, receiver) = mpsc::channel(2);
        let cancel = self.state.cancel.child_token();
        let finished = Arc::new(AtomicBool::new(false));
        let finished_notify = Arc::new(Notify::new());
        let state = self.state.clone();
        let task_cancel = cancel.clone();
        let task_finished = finished.clone();
        let task_notify = finished_notify.clone();
        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            run_sse(state, url, body, sender, task_cancel).await;
            task_finished.store(true, Ordering::Release);
            task_notify.notify_waiters();
        });
        Ok(NativeSseStream {
            receiver: Arc::new(Mutex::new(receiver)),
            cancel,
            finished,
            finished_notify,
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        state.closed.store(true, Ordering::Release);
        state.cancel.cancel();
        state.admission.close();
        state.connections.close();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                if state.active.load(Ordering::Acquire) == 0 {
                    return Ok(());
                }
                let notified = state.idle.notified();
                if state.active.load(Ordering::Acquire) == 0 {
                    return Ok(());
                }
                notified.await;
            }
        })
    }

    fn _limits(&self) -> (usize, usize) {
        (self.state.admission_limit, self.state.connection_limit)
    }
}

#[pyclass(module = "pygent._native")]
struct NativeSseStream {
    receiver: Arc<Mutex<mpsc::Receiver<StreamItem>>>,
    cancel: CancellationToken,
    finished: Arc<AtomicBool>,
    finished_notify: Arc<Notify>,
}

#[pymethods]
impl NativeSseStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = self.receiver.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match receiver.lock().await.recv().await {
                Some(item) => stream_item_to_py(item),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }

    fn close(&self) {
        self.cancel.cancel();
    }

    fn wait_closed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let finished = self.finished.clone();
        let notify = self.finished_notify.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                if finished.load(Ordering::Acquire) {
                    return Ok(());
                }
                let notified = notify.notified();
                if finished.load(Ordering::Acquire) {
                    return Ok(());
                }
                notified.await;
            }
        })
    }
}

fn stream_item_to_py(item: StreamItem) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| match item {
        StreamItem::Data(value) => Ok(("data", value).into_pyobject(py)?.unbind().into_any()),
        StreamItem::Status(value) => Ok(("status", value).into_pyobject(py)?.unbind().into_any()),
        StreamItem::Error(value) => Ok(("error", value).into_pyobject(py)?.unbind().into_any()),
    })
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    consumed: usize,
    scanned: usize,
    data: Vec<u8>,
    has_data: bool,
}

impl SseDecoder {
    fn push(&mut self, chunk: &[u8]) {
        // Compact once per network chunk, never once per line. Keep the scan
        // cursor across fragments so long lines are traversed only once.
        if self.consumed > 0 {
            self.buffer.drain(..self.consumed);
            self.scanned -= self.consumed;
            self.consumed = 0;
        }
        self.buffer.extend_from_slice(chunk);
    }

    fn next_event(&mut self) -> Option<Result<String, std::string::FromUtf8Error>> {
        while let Some(offset) = self.buffer[self.scanned..].iter().position(|b| *b == b'\n') {
            let end = self.scanned + offset;
            let line = &self.buffer[self.consumed..end];
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            self.consumed = end + 1;
            self.scanned = self.consumed;
            if line.is_empty() && self.has_data {
                self.has_data = false;
                return Some(String::from_utf8(std::mem::take(&mut self.data)));
            }
            if let Some(value) = line.strip_prefix(b"data:") {
                if self.has_data {
                    self.data.push(b'\n');
                }
                self.data
                    .extend_from_slice(value.strip_prefix(b" ").unwrap_or(value));
                self.has_data = true;
            }
        }
        self.scanned = self.buffer.len();
        None
    }
}

#[cfg(test)]
mod sse_tests {
    use super::SseDecoder;

    #[test]
    fn arbitrary_fragmentation_preserves_multiline_unicode_and_empty_data() {
        let input = ":comment\r\nevent: delta\r\ndata: 上海\r\ndata: world\r\n\r\ndata:\n\ndata: [DONE]\n\n";
        for chunk_size in 1..=input.len() {
            let mut decoder = SseDecoder::default();
            let mut events = Vec::new();
            for chunk in input.as_bytes().chunks(chunk_size) {
                decoder.push(chunk);
                while let Some(event) = decoder.next_event() {
                    events.push(event.unwrap());
                }
            }
            assert_eq!(events, ["上海\nworld", "", "[DONE]"]);
        }
    }

    #[test]
    fn long_fragmented_line_advances_scan_without_losing_partial_data() {
        let mut decoder = SseDecoder::default();
        decoder.push(b"data: ");
        assert!(decoder.next_event().is_none());
        for _ in 0..4096 {
            decoder.push(b"xxxxxxxxxxxxxxxx");
            assert!(decoder.next_event().is_none());
            assert_eq!(decoder.scanned, decoder.buffer.len());
        }
        decoder.push(b"\n\n");
        assert_eq!(decoder.next_event().unwrap().unwrap(), "x".repeat(65536));
        assert!(decoder.next_event().is_none());
    }

    #[test]
    fn invalid_utf8_is_reported_only_at_a_complete_event() {
        let mut decoder = SseDecoder::default();
        decoder.push(b"data: \xff\n");
        assert!(decoder.next_event().is_none());
        decoder.push(b"\n");
        assert!(decoder.next_event().unwrap().is_err());
    }
}

async fn run_sse(
    state: Arc<ClientState>,
    url: String,
    body: String,
    sender: mpsc::Sender<StreamItem>,
    cancel: CancellationToken,
) {
    let permit = match cancel
        .run_until_cancelled(state.admission.clone().acquire_owned())
        .await
    {
        Some(permit) => match permit {
            Ok(value) => value,
            Err(_) => return,
        },
        None => return,
    };
    let _permit = permit;
    let connection = match cancel
        .run_until_cancelled(state.connections.clone().acquire_owned())
        .await
    {
        Some(permit) => match permit {
            Ok(value) => value,
            Err(_) => return,
        },
        None => return,
    };
    let _connection = connection;
    let _active = ActiveGuard::new(state.clone());
    let response = match cancel
        .run_until_cancelled(
            state
                .client
                .post(url)
                .header(CONTENT_TYPE, "application/json")
                .header(ACCEPT, "text/event-stream")
                .header(CONNECTION, "keep-alive")
                .body(body)
                .send(),
        )
        .await
    {
        Some(result) => match result {
            Ok(value) => value,
            Err(error) => {
                let _ =
                    send_stream_item(&sender, &cancel, StreamItem::Error(error.to_string())).await;
                return;
            }
        },
        None => return,
    };
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let _ = send_stream_item(&sender, &cancel, StreamItem::Status(status)).await;
        return;
    }
    let mut stream = response.bytes_stream();
    let mut decoder = SseDecoder::default();
    loop {
        let chunk = match cancel.run_until_cancelled(stream.next()).await {
            Some(value) => value,
            None => return,
        };
        let Some(chunk) = chunk else { break };
        let chunk = match chunk {
            Ok(value) => value,
            Err(error) => {
                let _ =
                    send_stream_item(&sender, &cancel, StreamItem::Error(error.to_string())).await;
                return;
            }
        };
        decoder.push(&chunk);
        while let Some(payload) = decoder.next_event() {
            let payload = match payload {
                Ok(value) => value,
                Err(error) => {
                    let _ =
                        send_stream_item(&sender, &cancel, StreamItem::Error(error.to_string()))
                            .await;
                    return;
                }
            };
            let terminal = payload.trim() == "[DONE]";
            if terminal {
                let drain = async {
                    while let Some(chunk) = stream.next().await {
                        if chunk.is_err() {
                            break;
                        }
                    }
                };
                let _ = tokio::time::timeout(
                    Duration::from_millis(50),
                    cancel.run_until_cancelled(drain),
                )
                .await;
                let _ = send_stream_item(&sender, &cancel, StreamItem::Data(payload)).await;
                return;
            }
            if !send_stream_item(&sender, &cancel, StreamItem::Data(payload)).await {
                return;
            }
        }
    }
}

async fn send_stream_item(
    sender: &mpsc::Sender<StreamItem>,
    cancel: &CancellationToken,
    item: StreamItem,
) -> bool {
    matches!(
        cancel.run_until_cancelled(sender.send(item)).await,
        Some(Ok(()))
    )
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeHttpClient>()?;
    module.add_class::<NativeSseStream>()?;
    Ok(())
}
