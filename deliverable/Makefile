.PHONY: all clean help report beamer view-report view-beamer

# Main targets
all: report

report:
	latexmk tensygrid-report.tex

pdfserver:
	python3 -m http.server 40001

beamer:
	latexmk tensygrid-beamer.tex

view-report: report
	xdg-open tensygrid-report.pdf &

view-beamer: beamer
	xdg-open tensygrid-beamer.pdf &

clean:
	latexmk -c
	rm -f *.pdf

distclean: clean
	latexmk -C
	find . -name "*.aux" -delete
	find . -name "*.bbl" -delete
	find . -name "*.bcf" -delete
	find . -name "*.blg" -delete
	find . -name "*.fls" -delete
	find . -name "*.fdb_latexmk" -delete

help:
	@echo "Targets disponibles:"
	@echo "  make              - Compila el report principal (tensygrid-report.pdf)"
	@echo "  make report       - Compila el report"
	@echo "  make beamer       - Compila la presentació (tensygrid-beamer.pdf)"
	@echo "  make view-report  - Compila i obri el report"
	@echo "  make view-beamer  - Compila i obri la presentació"
	@echo "  make clean        - Neteja fitxers temporals"
	@echo "  make distclean    - Neteja tots els fitxers generats (inclosos PDFs)"
	@echo "  make help         - Mostra aquesta ajuda"
