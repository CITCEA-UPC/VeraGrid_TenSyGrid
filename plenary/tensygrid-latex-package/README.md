# TenSyGrid $\LaTeX$ Package (under development)
This package provides a style definition for typesetting documents in TenSyGrid design guidelines.

## Table of Contents
* [Support](#support)
* [Usage](#usage)
  * [Partner Acronyms](#partner-acronyms)
* [Attendance Certificates](#attendance-certificates)

## Support
The package currently only supports the following document classes:

+ [beamer (alpha state)](https://ctan.org/pkg/beamer) - LaTeX class for producing presentations and slides
+ [koma-script](https://ctan.org/pkg/koma-script) - A bundle of versatile classes and packages
  + [scrreprt (pre-alpha)](https://ctan.org/pkg/scrreprt) - Koma-Script "report" class
## Usage
This packages requires the `xelatex` $\LaTeX$ program that runs `xetex` $\TeX$ engine. This is because the regular `pdflatex` setup is not ideal for external font usage, i.e, the `Ralleway` font family for TenSyGrid.

### Partner Acronyms
The package contains predefined templates adapted to each TenSyGrid partner via an optional argument, e.g.,
```latex
\usepackage[IWES]{tensygrid}
```

A table with all the partner acronyms is given next:

| Acronym | Partner                                      |
|---------|----------------------------------------------|
| IWES    | Fraunhofer Institute for Wind Energy Systems |
| UPC     | Universitat Politècnica de Catalunya         |
| eRoots  | eRoots Analytics                             |
| HAW     | Hamburg University of Applied Sciences       |
| UM      | University of Malta                          |

## Attendance Certificates

Open [tensygrid-certificate.tex](tensygrid-certificate.tex) and edit the settings marked `EDIT HERE`: event, dates, location, signatory, and attendees. Replace all bracketed placeholders. The hackathon is an example; the template can be reused for other events.

Choose the signatory's institution with the partner option, for example `\usepackage[UPC]{tensygrid}` (see [Partner Acronyms](#partner-acronyms)). Its name appears beneath the signature; enter the person's name separately.

Keep one entry per attendee:

```latex
\Attendee{Luke Skywalker}
\Attendee{Jabba the Hutt}
\Attendee{Padmé Amidala}
```

Names support UTF-8 characters and LaTeX accents. Escape special characters, for example `\&` for `&`.

### Generate PDFs

Edit the existing `\CertificateBatchExport` and `\CertificateOnly` settings:

| Output | Batch Export | Only |
| --- | --- | --- |
| Combined PDF and individual PDFs (default) | `1` | Ignored |
| Combined PDF only | `0` | `0` |
| One attendee | `0` | List position: `1`, `2`, etc. |

In **VS Code**, open the package repository folder directly and press **Ctrl+Alt+B** with the certificate file active.

From a **terminal** in the package directory, batch export requires shell escape:

```sh
xelatex -shell-escape tensygrid-certificate.tex
```

For combined-only or single-attendee output, set batch export to `0`, then run this command twice:

```sh
xelatex -no-shell-escape tensygrid-certificate.tex
```

The main output is `tensygrid-certificate.pdf`. Batch export also writes `certificates/certificate-001.pdf`, `certificate-002.pdf`, etc., in attendee-list order. Keep the `certificates/` folder present. Matching files are overwritten; remove outdated exports when shortening or reordering the list.

### Overleaf

Upload the package contents, including `config`, `fonts`, and `koma`, and ensure a `certificates/` folder exists (the repository includes `certificates/.gitkeep`). Select `tensygrid-certificate.tex` as the main document and **XeLaTeX** as the compiler, then **Recompile**.

For combined-only or single-attendee output, set batch export to `0` and choose `\CertificateOnly` using the table above. Download the main PDF after recompiling.

Overleaf batch export is not yet verified. It requires unrestricted shell escape; generated individual PDFs should be available under **Logs and output files > Other logs and output files**. Keep the main filename unchanged and download generated PDFs before clearing the compilation cache.

For large lists, first download the complete combined PDF with both settings at `0`. Then export smaller attendee subsets in batch mode if compilation times out. Download each subset before exporting the next; numbered filenames are reused. Use **Recompile from scratch** to clear outdated exports between subsets.

### Signature

Leave `\CertificateSignatureFile` empty to sign by hand or in a PDF editor. To include a signature image, change the existing setting to a tightly cropped transparent PNG or PDF, relative to the package directory:

```latex
\newcommand{\CertificateSignatureFile}{signature.png}
```

A signature image is not a cryptographic PDF signature. Share only the finished PDFs, not the signature image.
