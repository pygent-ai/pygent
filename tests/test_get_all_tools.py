"""
测试 tool_class 装饰类上的 get_all_tools() 方法。
用 FileToolkits 验证：应返回工具实例列表，而非 schema 字典。
"""
from pygent.toolkits.file_operations import FileToolkits
from pygent.module.tool import BaseTool


def main():
    tk = FileToolkits(session_id="test", workspace_root=".")
    result = tk.get_all_tools()

    # 预期：get_all_tools 返回 List[BaseTool]，可迭代且每项为 BaseTool
    print("get_all_tools() 返回类型:", type(result))
    print("是否为 list:", isinstance(result, list))
    print("是否为 dict:", isinstance(result, dict))

    if isinstance(result, dict):
        print("字典的 keys:", list(result.keys()))
        # 错误实现会返回 get_all_schemas() -> {"tools": {...}, "categories": {...}}
        if "tools" in result:
            print("  -> 实际返回的是 schema（get_all_schemas），不是工具实例列表")
        for k in result:
            v = result[k]
            print(f"  {k}: type={type(v).__name__}")
    elif isinstance(result, list):
        print("列表长度:", len(result))
        for i, item in enumerate(result[:3]):
            print(f"  [{i}] type={type(item).__name__}, is BaseTool={isinstance(item, BaseTool)}")
            if isinstance(item, BaseTool):
                print(f"       name={item.metadata.data.get('name')}")


if __name__ == "__main__":
    main()
