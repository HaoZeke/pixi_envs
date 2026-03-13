;;; export.el --- Export orgmode docs to RST for Sphinx -*- lexical-binding: t; -*-

;;; Commentary:
;; Export NEB Orchestrator orgmode documentation to RST for Sphinx build.
;; Run with: emacs --batch --load export.el -f neb-export-all

;;; Code:

(require 'ox-rst)

(setq neb-org-dir (file-name-directory (buffer-file-name)))
(setq neb-rst-dir (expand-file-name "source" neb-org-dir))

(defun neb-export-file (org-file rst-file)
  "Export ORG-FILE to RST-FILE."
  (find-file org-file)
  (org-rst-export-to-rst nil nil nil nil nil t)
  (let ((rst-output (concat (file-name-sans-extension org-file) ".rst")))
    (when (file-exists-p rst-output)
      (copy-file rst-output rst-file t)
      (delete-file rst-output)))
  (kill-buffer))

(defun neb-export-all ()
  "Export all orgmode files to RST."
  (interactive)
  (let ((files '(
                 ("orgmode/index.org" "source/index.rst")
                 ("orgmode/quickstart.org" "source/quickstart.rst")
                 ("orgmode/devnotes.org" "source/devnotes.rst")
                 )))
    (dolist (file-pair files)
      (let ((org-file (expand-file-name (car file-pair) neb-org-dir))
            (rst-file (expand-file-name (cadr file-pair) neb-org-dir)))
        (when (file-exists-p org-file)
          (message "Exporting %s to %s" org-file rst-file)
          (neb-export-file org-file rst-file))))))

(provide 'export)
;;; export.el ends here
