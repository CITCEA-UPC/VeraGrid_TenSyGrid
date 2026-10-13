# Deliverable D4.1 — TenSyGrid Toolbox: Use Cases Identification and Description

Deliverable D4.1 report of the **TenSyGrid** project (CETP-FP-2023-00138), under WP4. The document identifies and describes the use cases that the *TenSyGrid Toolbox* must support to assist network operators in assessing the small-signal stability of converter-dominated power grids.

## Authors

- **Alexandre Gràcia Calvo** — PhD Student, UPC
- **Carlos Herrera Vázquez** — PhD Student, UPC
- **Dr. Eduardo Prieto Araujo** — Director, UPC

Universitat Politècnica de Catalunya (UPC)

## Building

A LaTeX distribution with `lualatex`, `latexmk` and the `tensygrid` package is required.

```bash
make            # Build the report (tensygrid-report.pdf)
make report     # Same as above
make beamer     # Build the presentation (tensygrid-beamer.pdf)

make view-report  # Build and open the report
make view-beamer  # Build and open the presentation

make help       # Show all available targets
```

### Cleanup

```bash
make clean       # Remove temporary files
make distclean   # Remove all generated files, including PDFs
```

## Repository structure

```
.
├── Makefile
├── README.md
├── .latexmkrc
├── .gitlab-ci.yml
├── .gitignore
├── LICENSE.md
├── tensygrid-report.tex       # Main document (report)
├── tensygrid-beamer.tex       # Presentation
├── tensygrid.sty              # Document style
├── IEEEtrantools.sty
├── ref.bib                    # Bibliography
├── beamer/                    # Beamer theme files
├── config/                    # Logos and partner config files
├── docs/                      # Project proposal and Gantt chart
├── fonts/                     # Raleway font files
├── koma/                      # KOMA-Script customizations
├── pictures/                  # Figures and screenshots
│   └── loads/
├── Chapters/
│   ├── introduction.tex
│   ├── use_cases.tex
│   ├── conclusion.tex
│   └── examples/
│       ├── lines.tex
│       ├── loads.tex
│       ├── transformers.tex
│       ├── converters.tex
│       ├── hvdc.tex
│       └── facts.tex
└── .vscode/
```

## License

Internal document of the TenSyGrid project. Classification: **Public**.
