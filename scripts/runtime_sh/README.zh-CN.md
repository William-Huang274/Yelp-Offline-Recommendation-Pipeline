# 运行时 Shell 启动器

[English](./README.md) | [中文](./README.zh-CN.md)

这里集中存放当前 `Stage09 -> Stage11` 主链使用的长名 shell 启动器。

这些文件原先直接放在 `scripts/` 根目录下。现在统一收进本目录，目的是让
根目录脚本表面更聚焦于 Python 主脚本、审计脚本和对外 wrapper。

对外展示、文档引用和演示入口统一使用 [../launchers/](../launchers)。

如果需要查看 wrapper 最终转发到哪一个完整 shell 启动器，则到这个目录下查看。
