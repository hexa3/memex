# Security Policy

## Supported versions

Security fixes are issued for the latest minor release.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub Security Advisories.

Do not include sensitive user databases in public issues. If a malformed database or import file triggers the issue, provide a minimal synthetic reproduction instead.

## Security posture

memex core is local-first and does not collect telemetry. The REST server binds to `127.0.0.1` by default. Users who bind to public interfaces should put the server behind their own authentication and network controls.

Sync is disabled by default. Optional sync payloads must be encrypted before leaving the device; the built-in sync codec uses authenticated encryption through the `sync` extra. Browser extension captures stay in browser-local storage unless sync is explicitly enabled.

Sensitive data handling rules:

- Do not commit exported memory databases or browser captures.
- Do not store sync passphrases in plaintext.
- Treat remote sync transports as untrusted.
- Prefer synthetic fixtures in issues and tests.
