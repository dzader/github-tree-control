# GitHub Folder Viewed

A small Chrome extension that adds a checkbox to every folder in a GitHub pull
request's **Files changed** tree.

- Check a folder to mark every changed file beneath it as viewed.
- Uncheck a fully viewed folder to mark every file beneath it as not viewed.
- Collapse a folder to also collapse every expanded file diff beneath it without
  changing any file's viewed state.
- Expand a folder to reopen every collapsed file diff beneath it.
- Parent folders use an indeterminate state when only some files are viewed.
- Folder Viewed checkboxes continue to work while their folder is collapsed.
- Nested and GitHub-compressed folder paths are supported.
- GitHub's own **Viewed** buttons are clicked, so the normal GitHub behavior and
  review state remain the source of truth.

## Install locally

1. Open `chrome://extensions` in Chrome.
2. Turn on **Developer mode**.
3. Click **Load unpacked**.
4. Select this project folder.
5. Open or refresh a pull request's **Files changed** page.

No GitHub token, extension settings, or additional permissions are required.

## Updating

After changing the extension files, click **Reload** for the extension on
`chrome://extensions`, then refresh the GitHub pull request page.
