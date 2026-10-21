.PHONY: all clean distclean help view

# Main target
all: hackathon

hackathon:
	latexmk hackathon.tex

view: hackathon
	xdg-open hackathon.pdf &

clean:
	latexmk -c
	rm -f *.pdf *.png

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
	@echo "  make              - Compila el document (hackathon.pdf)"
	@echo "  make hackathon    - Compila el document"
	@echo "  make view         - Compila i obri el PDF"
	@echo "  make clean        - Neteja fitxers temporals i PDFs"
	@echo "  make distclean    - Neteja tots els fitxers generats"
	@echo "  make help         - Mostra aquesta ajuda"
