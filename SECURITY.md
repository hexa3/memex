# Security Policy

## Supported versions

Security fixes are issued for the latest minor release.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub Security Advisories.

Do not include sensitive user databases in public issues. If a malformed database or import file triggers the issue, provide a minimal synthetic reproduction instead.

## Security posture

memex core is local-first and does not collect telemetry. The REST server binds to `127.0.0.1` by default. Users who bind to public interfaces should put the server behind their own authentication and network controls.
