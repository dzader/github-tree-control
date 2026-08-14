(() => {
  "use strict";

  const TREE_SELECTOR = '[role="tree"][aria-label="File Tree"]';
  const FOLDER_SELECTOR = 'li[role="treeitem"][aria-expanded]';
  const FILE_LINK_SELECTOR = 'li[role="treeitem"] a[href^="#diff-"]';
  const VIEWED_BUTTON_SELECTOR =
    'button[aria-label="Viewed"], button[aria-label="Not Viewed"]';
  const CHECKBOX_CLASS = "gh-folder-viewed-checkbox";
  const ROW_CLASS = "gh-folder-viewed-row";
  const STATUS_CLASS = "gh-folder-viewed-status";
  const UPDATE_DELAY_MS = 100;
  const BUTTON_TIMEOUT_MS = 5000;

  let updateTimer;
  let operationInProgress = false;
  const folderTargetCache = new Map();

  const getFolderPath = (folder) => folder.id || "folder";
  const getFolderCacheKey = (folder) =>
    `${window.location.pathname}:${getFolderPath(folder)}`;

  const getFileTargets = (folder) => {
    const group = folder.querySelector(':scope > ul[role="group"]');

    if (!group) {
      return [];
    }

    return [...group.querySelectorAll(FILE_LINK_SELECTOR)]
      .map((link) => ({
        diffId: link.getAttribute("href")?.slice(1),
        path: link.closest('li[role="treeitem"]')?.id,
      }))
      .filter(({ diffId, path }) => diffId && path);
  };

  const rememberFileTargets = (folder) => {
    const targets = getFileTargets(folder);

    if (targets.length > 0) {
      folderTargetCache.set(getFolderCacheKey(folder), targets);
    }

    return targets;
  };

  const getDiffPath = (region) => {
    const dataPath = region
      .querySelector("[data-file-path]")
      ?.getAttribute("data-file-path");

    if (dataPath) {
      return dataPath;
    }

    return (
      region
        .querySelector("h3 code")
        ?.textContent?.replace(/[\u200e\u200f\u202a-\u202e\u2066-\u2069]/g, "")
        .trim() || ""
    );
  };

  const reconstructFileTargets = (folder) => {
    const folderPrefix = `${getFolderPath(folder)}/`;

    return [...document.querySelectorAll('[id^="diff-"][role="region"]')]
      .map((region) => ({
        diffId: region.id,
        path: getDiffPath(region),
      }))
      .filter(({ diffId, path }) => diffId && path.startsWith(folderPrefix));
  };

  const getRememberedFileTargets = (folder) => {
    const targets = getFileTargets(folder);

    if (targets.length > 0) {
      folderTargetCache.set(getFolderCacheKey(folder), targets);
      return targets;
    }

    const cachedTargets = folderTargetCache.get(getFolderCacheKey(folder));

    if (cachedTargets?.length) {
      return cachedTargets;
    }

    const reconstructedTargets = reconstructFileTargets(folder);

    if (reconstructedTargets.length > 0) {
      folderTargetCache.set(getFolderCacheKey(folder), reconstructedTargets);
    }

    return reconstructedTargets;
  };

  const getAccessibleLabel = (button) => {
    const label = button.getAttribute("aria-label");

    if (label) {
      return label.trim();
    }

    return (button.getAttribute("aria-labelledby") || "")
      .split(/\s+/)
      .map((id) => document.getElementById(id)?.textContent?.trim())
      .filter(Boolean)
      .join(" ");
  };

  const getFileToggleButton = (diffId) => {
    const header = document
      .getElementById(diffId)
      ?.querySelector("[data-diff-header-wrapper]");

    if (!header) {
      return null;
    }

    return (
      [...header.querySelectorAll("button")].find((button) =>
        ["Collapse file", "Expand file"].includes(getAccessibleLabel(button)),
      ) || null
    );
  };

  const setFolderFilesExpanded = (folder, expanded) => {
    const targets = getRememberedFileTargets(folder);
    const currentControlLabel = expanded ? "Expand file" : "Collapse file";
    const fileButtons = targets
      .map(({ diffId }) => getFileToggleButton(diffId))
      .filter(
        (button) =>
          button && getAccessibleLabel(button) === currentControlLabel,
      );

    fileButtons.forEach((button) => button.click());

    if (fileButtons.length > 0) {
      setStatus(
        `${expanded ? "Expanded" : "Collapsed"} ${fileButtons.length} files in ${getFolderPath(folder)}.`,
      );
    }
  };

  const getViewedButton = (diffId) => {
    const region = document.getElementById(diffId);
    return region?.querySelector(VIEWED_BUTTON_SELECTOR) || null;
  };

  const isViewed = (button) =>
    button?.getAttribute("aria-pressed") === "true" ||
    button?.getAttribute("aria-label") === "Viewed";

  const getStatusRegion = (tree) => {
    const existing = document.querySelector(`.${STATUS_CLASS}`);

    if (existing) {
      return existing;
    }

    const status = document.createElement("div");
    status.className = STATUS_CLASS;
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    tree.insertAdjacentElement("afterend", status);
    return status;
  };

  const setStatus = (message) => {
    const tree = document.querySelector(TREE_SELECTOR);

    if (tree) {
      getStatusRegion(tree).textContent = message;
    }
  };

  const updateCheckbox = (folder, checkbox) => {
    const targets = getRememberedFileTargets(folder);
    const viewedCount = targets.filter(({ diffId }) =>
      isViewed(getViewedButton(diffId)),
    ).length;
    const allViewed = targets.length > 0 && viewedCount === targets.length;
    const folderPath = getFolderPath(folder);

    checkbox.checked = allViewed;
    checkbox.indeterminate = viewedCount > 0 && !allViewed;
    checkbox.dataset.busy = String(operationInProgress);
    checkbox.disabled = operationInProgress || targets.length === 0;
    const label = allViewed
      ? `Mark all files in ${folderPath} as not viewed`
      : `Mark all files in ${folderPath} as viewed`;

    if (checkbox.getAttribute("aria-label") !== label) {
      checkbox.setAttribute("aria-label", label);
    }
    checkbox.title = `${viewedCount} of ${targets.length} files viewed`;
  };

  const waitForViewedState = (diffId, viewed) =>
    new Promise((resolve, reject) => {
      const startedAt = Date.now();

      const check = () => {
        const button = getViewedButton(diffId);

        if (button && isViewed(button) === viewed) {
          resolve();
          return;
        }

        if (Date.now() - startedAt >= BUTTON_TIMEOUT_MS) {
          reject(new Error(`GitHub did not update ${diffId}`));
          return;
        }

        window.setTimeout(check, UPDATE_DELAY_MS);
      };

      check();
    });

  const setAllControlsDisabled = (disabled) => {
    document.querySelectorAll(`.${CHECKBOX_CLASS}`).forEach((checkbox) => {
      checkbox.dataset.busy = String(disabled);
      checkbox.disabled = disabled;
    });
  };

  const updateAllCheckboxes = () => {
    document.querySelectorAll(`.${CHECKBOX_CLASS}`).forEach((checkbox) => {
      const folder = checkbox.closest(FOLDER_SELECTOR);

      if (folder) {
        updateCheckbox(folder, checkbox);
      }
    });
  };

  const setFolderViewed = async (folder, viewed) => {
    if (operationInProgress) {
      return;
    }

    operationInProgress = true;
    setAllControlsDisabled(true);

    const targets = getRememberedFileTargets(folder);
    const folderPath = getFolderPath(folder);
    const failures = [];
    let changedCount = 0;

    try {
      for (const { diffId, path } of targets) {
        const button = getViewedButton(diffId);

        if (!button) {
          failures.push(path);
          continue;
        }

        if (isViewed(button) === viewed) {
          continue;
        }

        try {
          button.click();
          await waitForViewedState(diffId, viewed);
          changedCount += 1;
        } catch {
          failures.push(path);
        }
      }
    } finally {
      operationInProgress = false;
      updateAllCheckboxes();
    }

    if (failures.length > 0) {
      setStatus(
        `Updated ${changedCount} files in ${folderPath}. ${failures.length} files could not be updated.`,
      );
      return;
    }

    setStatus(
      `${viewed ? "Marked" : "Unmarked"} ${targets.length} files in ${folderPath} as viewed.`,
    );
  };

  const addCheckbox = (folder) => {
    const row = folder.querySelector(":scope > div");

    if (!row || row.querySelector(`.${CHECKBOX_CLASS}`)) {
      return;
    }

    row.classList.add(ROW_CLASS);

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = CHECKBOX_CLASS;

    ["pointerdown", "mousedown", "click", "keydown"].forEach((eventName) => {
      checkbox.addEventListener(eventName, (event) => event.stopPropagation());
    });

    checkbox.addEventListener("change", () => {
      void setFolderViewed(folder, checkbox.checked);
    });

    row.append(checkbox);
    updateCheckbox(folder, checkbox);
  };

  const enhanceFileTree = () => {
    const tree = document.querySelector(TREE_SELECTOR);

    if (!tree) {
      return;
    }

    getStatusRegion(tree);
    tree.querySelectorAll(FOLDER_SELECTOR).forEach((folder) => {
      rememberFileTargets(folder);
      addCheckbox(folder);
    });
    updateAllCheckboxes();
  };

  const scheduleEnhancement = () => {
    window.clearTimeout(updateTimer);
    updateTimer = window.setTimeout(enhanceFileTree, UPDATE_DELAY_MS);
  };

  const observer = new MutationObserver((mutations) => {
    mutations
      .filter(
        (mutation) =>
          mutation.type === "attributes" &&
          mutation.attributeName === "aria-expanded" &&
          mutation.target.matches?.(FOLDER_SELECTOR) &&
          ["true", "false"].includes(mutation.oldValue) &&
          ["true", "false"].includes(
            mutation.target.getAttribute("aria-expanded"),
          ) &&
          mutation.oldValue !== mutation.target.getAttribute("aria-expanded"),
      )
      .forEach((mutation) =>
        setFolderFilesExpanded(
          mutation.target,
          mutation.target.getAttribute("aria-expanded") === "true",
        ),
      );

    scheduleEnhancement();
  });
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeOldValue: true,
    attributeFilter: ["aria-label", "aria-pressed", "aria-expanded"],
  });

  enhanceFileTree();
})();
