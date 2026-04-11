# Runtime Shell Launchers

[English](./README.md) | [中文](./README.zh-CN.md)

This directory holds the long-form shell launchers for the active
`Stage09 -> Stage11` runtime path.

They used to live directly under `scripts/`. They are now grouped here so that
the root script surface stays focused on Python entry scripts, audit helpers,
and the outward-facing wrapper layer.

Use [../launchers/](../launchers) as the public entry surface.

Use this directory when you need the underlying full shell launcher that a
wrapper forwards to.
