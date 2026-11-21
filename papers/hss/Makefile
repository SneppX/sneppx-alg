.PHONY: all pdf clean arxiv

TEX = main.tex
PDF = main.pdf
AUX = main.aux main.bbl main.blg main.log main.out main.toc
ARXIV_FILES = main.tex references.bib IEEEtran.cls

all: pdf

pdf:
	pdflatex $(TEX)
	bibtex main
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f $(AUX) $(PDF)
	rm -rf arxiv/

arxiv: pdf
	mkdir -p arxiv
	cp $(ARXIV_FILES) arxiv/
	cp $(PDF) arxiv/
	tar -czf arxiv.tar.gz arxiv/
	rm -rf arxiv/
