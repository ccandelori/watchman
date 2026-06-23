# Connecting Claude Code to the Asana MCP Server

This documents the exact steps (and the dead ends) we hit wiring Claude Code up to
the Asana MCP server so it can read/create tasks on our **Aegis** board. If you are
setting this up on a new machine, follow the "Working setup" section and skip the
troubleshooting history unless something breaks.

## TL;DR (working setup)

1. Create your own OAuth app in the Asana developer console — Asana's production MCP
   server does **not** support dynamic client registration, so each user needs a
   pre-registered client ID/secret.
2. Point Claude Code at the **V2 streamable-HTTP endpoint**, not the retired beta SSE one.
3. Authenticate through the browser.

```bash
claude mcp add --transport http \
  --client-id YOUR_CLIENT_ID \
  --client-secret \
  --callback-port 8080 \
  asana https://mcp.asana.com/v2/mcp
```

Then run `/mcp` → **asana** → **Authenticate** and complete the browser sign-in.
A successful run reports `Authentication successful. Connected to asana.`

## Step-by-step

### 1. Create an Asana MCP OAuth app

1. Go to the [Asana developer console → My Apps](https://app.asana.com/0/my-apps).
2. Create a new app (an **MCP app**). Tokens issued for MCP apps only work with the
   MCP server — a regular Asana API app's tokens will **not** work.
3. Under **OAuth**, set the **Redirect URL** to exactly:

   ```
   http://localhost:8080/callback
   ```

4. **Save the app.** (See gotcha below — an unsaved redirect URL silently fails.)
5. Copy the generated **Client ID** and **Client Secret**.

### 2. Register the server with Claude Code

Run the command from the TL;DR, replacing `YOUR_CLIENT_ID`. The `--client-secret`
flag prompts for the secret interactively (input is hidden); it is stored in the
macOS keychain, not in plaintext. `--callback-port 8080` makes Claude Code use the
fixed redirect URI `http://localhost:8080/callback`, which must match the app.

### 3. Authenticate

Run `/mcp`, select **asana**, choose **Authenticate**, and complete the browser
"Allow access" flow. The status should flip to **✔ Connected**.

### 4. Verify

```bash
claude mcp list   # asana: https://mcp.asana.com/v2/mcp (HTTP) - ✔ Connected
```

## Troubleshooting — the errors we actually hit, in order

We started from the (then-documented) beta endpoint and worked through a chain of
failures. Each fix exposed the next problem:

| Symptom | Cause | Fix |
| --- | --- | --- |
| `HTTP 404 at https://mcp.asana.com/sse` after the browser auth succeeded | The beta **SSE** endpoint was retired (shutdown ~May 2026). OAuth negotiates against a separate service, so the failure only surfaces as a post-auth 404. | Use the V2 endpoint `https://mcp.asana.com/v2/mcp` (streamable HTTP). |
| `Incompatible auth server: does not support dynamic client registration` | Asana's V2 server does **not** support dynamic client registration (DCR / RFC 7591); Claude Code's default flow tries to auto-register a client. | Create your own OAuth app and pass `--client-id` / `--client-secret`. |
| `invalid_request: The redirect_uri parameter does not match a valid url for the application.` | The redirect URI Claude Code sends (`http://localhost:8080/callback`, from `--callback-port 8080`) didn't match what was registered on the Asana app. | Make the app's redirect URL match **character-for-character** — and make sure it is actually **saved**. |
| `MCP server asana already exists in local config` | A stale `asana` entry from a previous attempt. | `claude mcp remove asana` before re-adding. |

### redirect_uri must match exactly

When you see the `invalid_request` error, read the `redirect_uri=...` parameter
out of the browser's address bar (URL-decoded it is `http://localhost:8080/callback`)
and make the Asana app's redirect URL identical. Things that silently break the match:

- a **trailing slash** (`.../callback/` ≠ `.../callback`)
- `https://` instead of `http://` (use `http` for localhost)
- a different port than `--callback-port`
- an empty path or `/` instead of `/callback`
- **not clicking Save** in the console (this was our actual final blocker — auth in
  the browser succeeded, but the unsaved redirect URL was rejected)
- using a Client ID from a *different* app than the one where you set the redirect URL

## Notes / gotchas

- **Transport:** the V2 server uses **Streamable HTTP** (`--transport http`), not SSE.
- **Per-user apps:** because DCR is unsupported, every teammate registers their own
  OAuth app and runs their own `claude mcp add`. There is no shared client.
- **Token scope:** MCP tokens only work with the MCP server, not the standard Asana API.
- **Aegis project reference:** project name `Aegis`, workspace `YourUrbit.com`.

## References

- [Using Asana's MCP Server](https://developers.asana.com/docs/using-asanas-mcp-server)
- [Integrating with Asana's MCP Server](https://developers.asana.com/docs/integrating-with-asanas-mcp-server)
- [Connecting MCP clients to Asana's V2 server (Claude Code section)](https://developers.asana.com/docs/connecting-mcp-clients-to-asanas-v2-server)
