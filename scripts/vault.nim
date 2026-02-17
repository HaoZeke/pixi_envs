## Encrypted vault for pixi_envs.
##
## Stores sensitive files with opaque filenames and encrypted contents
## using GPG. Both filenames and contents are hidden in git.
##
## Usage:
##   nim r scripts/vault.nim <seal|unseal|add|rm|list|status> [path]

import std/[os, osproc, strutils, strformat, streams, terminal, sysrand]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

proc run(cmd: string, workDir = ""): tuple[output: string, exitCode: int] =
  execCmdEx(cmd, workingDir = if workDir.len > 0: workDir else: getCurrentDir())

proc runOrDie(cmd: string, workDir = ""): string =
  let (output, code) = run(cmd, workDir)
  if code != 0:
    stderr.writeLine &"FATAL: command failed (exit {code}):\n  {cmd}\n{output}"
    quit 1
  result = output.strip()

proc banner(msg: string) =
  let w = terminalWidth()
  let line = "─".repeat(min(w, 72))
  echo ""
  echo line
  styledEcho fgCyan, styleBright, "  ", msg
  echo line

proc repoRoot(): string =
  runOrDie("git rev-parse --show-toplevel")

# ---------------------------------------------------------------------------
# GPG operations
# ---------------------------------------------------------------------------

const GpgRecipient = "9CCCE36402CB49A6"

proc gpgEncrypt(inPath, outPath: string) =
  discard runOrDie(&"gpg --batch --yes --quiet --trust-model always " &
    &"-e -r {GpgRecipient} --set-filename \"\" -o {outPath.quoteShell} {inPath.quoteShell}")

proc gpgDecrypt(inPath, outPath: string) =
  discard runOrDie(&"gpg --batch --yes --quiet -d -o {outPath.quoteShell} {inPath.quoteShell}")

proc gpgDecryptToStdout(inPath: string): string =
  runOrDie(&"gpg --batch --yes --quiet -d {inPath.quoteShell}")

# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

proc sha256sum(path: string): string =
  ## Returns hex SHA-256 digest of a file.
  let res = runOrDie(&"sha256sum {path.quoteShell}")
  result = res.split(' ')[0]


# ---------------------------------------------------------------------------
# Manifest operations
# ---------------------------------------------------------------------------

type VaultEntry = tuple[id, path: string]

proc genId(): string =
  ## 16-char random hex via cryptographic randomness.
  var buf: array[8, byte]
  doAssert urandom(buf)
  for b in buf:
    result.add(b.toHex(2).toLowerAscii())

proc vaultDir(repo: string): string =
  repo / ".vault"

proc loadManifest(repo: string): seq[VaultEntry] =
  let enc = vaultDir(repo) / "manifest.gpg"
  if not fileExists(enc):
    return @[]
  let plain = gpgDecryptToStdout(enc)
  for line in plain.splitLines:
    let stripped = line.strip()
    if stripped.len == 0 or stripped.startsWith("#"):
      continue
    let parts = stripped.split('\t', maxsplit = 1)
    if parts.len == 2:
      result.add((parts[0], parts[1]))

proc saveManifest(repo: string, entries: seq[VaultEntry]) =
  let plainPath = vaultDir(repo) / ".manifest.plain"
  let encPath = vaultDir(repo) / "manifest.gpg"
  var content = "# vault-manifest-v1\n"
  for e in entries:
    content.add(&"{e.id}\t{e.path}\n")
  writeFile(plainPath, content)
  gpgEncrypt(plainPath, encPath)
  removeFile(plainPath)

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

proc unseal(repo: string) =
  let entries = loadManifest(repo)
  if entries.len == 0:
    echo "vault is empty"
    return

  banner("Unsealing vault ...")

  # Launch all GPG decrypts in parallel
  var procs: seq[(VaultEntry, Process)] = @[]
  for e in entries:
    let inPath = vaultDir(repo) / &"{e.id}.gpg"
    let outPath = repo / e.path
    if not fileExists(inPath):
      stderr.writeLine &"FATAL: vault blob missing: {inPath}"
      quit 1
    createDir(outPath.parentDir)
    let p = startProcess("/bin/bash",
      args = ["-c", &"gpg --batch --yes --quiet -d -o {outPath.quoteShell} {inPath.quoteShell}"],
      options = {poUsePath, poStdErrToStdOut})
    procs.add((e, p))

  # Collect results
  for (e, p) in procs:
    let output = p.outputStream.readAll()
    let code = p.waitForExit()
    p.close()
    if code != 0:
      stderr.writeLine &"FATAL: failed to unseal {e.path}\n{output}"
      quit 1
    # chmod 600
    setFilePermissions(repo / e.path, {fpUserRead, fpUserWrite})
    echo &"  {e.path}"

  echo &"\nUnsealed {entries.len} file(s)."

proc seal(repo: string) =
  let entries = loadManifest(repo)
  if entries.len == 0:
    echo "vault is empty"
    return

  banner("Sealing vault ...")

  # Verify all plaintext files exist first
  for e in entries:
    let src = repo / e.path
    if not fileExists(src):
      stderr.writeLine &"FATAL: plaintext missing: {src}"
      stderr.writeLine "  Run 'vault unseal' first, or 'vault rm' to remove the entry."
      quit 1

  # Launch all GPG encrypts in parallel
  var procs: seq[(VaultEntry, Process)] = @[]
  for e in entries:
    let inPath = repo / e.path
    let outPath = vaultDir(repo) / &"{e.id}.gpg"
    let p = startProcess("/bin/bash",
      args = ["-c", &"gpg --batch --yes --quiet --trust-model always " &
        &"-e -r {GpgRecipient} --set-filename \"\" -o {outPath.quoteShell} {inPath.quoteShell}"],
      options = {poUsePath, poStdErrToStdOut})
    procs.add((e, p))

  # Collect results
  for (e, p) in procs:
    let output = p.outputStream.readAll()
    let code = p.waitForExit()
    p.close()
    if code != 0:
      stderr.writeLine &"FATAL: failed to seal {e.path}\n{output}"
      quit 1
    echo &"  {e.path}"

  # Re-encrypt manifest
  saveManifest(repo, entries)
  echo &"\nSealed {entries.len} file(s)."

proc add(repo, path: string) =
  let relPath = if path.isAbsolute: relativePath(path, repo) else: path
  let absPath = if path.isAbsolute: path else: repo / path

  if not fileExists(absPath):
    stderr.writeLine &"FATAL: file not found: {absPath}"
    quit 1

  # Check for duplicates
  var entries = loadManifest(repo)
  for e in entries:
    if e.path == relPath:
      stderr.writeLine &"Already in vault: {relPath}"
      quit 1

  # Warn if not gitignored
  let (_, gitCheckCode) = run(&"git check-ignore -q {relPath.quoteShell}", repo)
  if gitCheckCode != 0:
    stderr.writeLine &"WARNING: {relPath} is NOT gitignored — add it to .gitignore"

  let id = genId()
  let outPath = vaultDir(repo) / &"{id}.gpg"

  banner(&"Adding {relPath} to vault ...")
  createDir(vaultDir(repo))
  gpgEncrypt(absPath, outPath)
  entries.add((id, relPath))
  saveManifest(repo, entries)
  echo &"  id:   {id}"
  echo &"  path: {relPath}"
  echo &"  blob: .vault/{id}.gpg"

proc remove(repo, path: string) =
  let relPath = if path.isAbsolute: relativePath(path, repo) else: path

  var entries = loadManifest(repo)
  var found = false
  var newEntries: seq[VaultEntry] = @[]
  for e in entries:
    if e.path == relPath:
      found = true
      let blobPath = vaultDir(repo) / &"{e.id}.gpg"
      if fileExists(blobPath):
        removeFile(blobPath)
        echo &"  Removed .vault/{e.id}.gpg"
      echo &"  Removed manifest entry: {relPath}"
    else:
      newEntries.add(e)

  if not found:
    stderr.writeLine &"Not in vault: {relPath}"
    quit 1

  saveManifest(repo, newEntries)
  echo "  (local plaintext file NOT deleted)"

proc list(repo: string) =
  let entries = loadManifest(repo)
  if entries.len == 0:
    echo "vault is empty"
    return
  for e in entries:
    echo &"  {e.id}  {e.path}"

proc status(repo: string) =
  let entries = loadManifest(repo)
  if entries.len == 0:
    echo "vault is empty"
    return

  banner("Vault status")
  for e in entries:
    let localPath = repo / e.path
    let blobPath = vaultDir(repo) / &"{e.id}.gpg"

    if not fileExists(localPath):
      styledEcho fgYellow, &"  [missing]   {e.path}"
      continue

    if not fileExists(blobPath):
      styledEcho fgRed, &"  [no-blob]   {e.path}"
      continue

    # Compare SHA-256 of local file vs decrypted vault blob (via temp file)
    let localHash = sha256sum(localPath)
    let tmpPath = vaultDir(repo) / &".status-tmp-{e.id}"
    gpgDecrypt(blobPath, tmpPath)
    let vaultHash = sha256sum(tmpPath)
    removeFile(tmpPath)

    if localHash == vaultHash:
      styledEcho fgGreen, &"  [in-sync]   {e.path}"
    else:
      styledEcho fgRed, &"  [modified]  {e.path}"

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

proc main() =
  let repo = repoRoot()
  if paramCount() < 1:
    stderr.writeLine "usage: vault <seal|unseal|add|rm|list|status> [path]"
    quit 1
  let cmd = paramStr(1)
  case cmd
  of "seal":
    seal(repo)
  of "unseal":
    unseal(repo)
  of "add":
    if paramCount() < 2:
      stderr.writeLine "usage: vault add <path>"
      quit 1
    add(repo, paramStr(2))
  of "rm":
    if paramCount() < 2:
      stderr.writeLine "usage: vault rm <path>"
      quit 1
    remove(repo, paramStr(2))
  of "list":
    list(repo)
  of "status":
    status(repo)
  else:
    stderr.writeLine &"unknown command: {cmd}"
    quit 1

when isMainModule:
  main()
