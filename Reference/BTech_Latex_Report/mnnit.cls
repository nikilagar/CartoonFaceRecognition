% MNNIT style for M.Tech and Ph.D Thesis
% -- morphed from the stanford style with numerous localizations!

%----------------------------- identification ---------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mnnit}[2005/07/01 v1.0 mnnit phd and mtech thesis report]
%----------------------------- initial code -----------------------------------

%--------------------------- declaration of options ---------------------------
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
%--------------------------- execution of options -----------------------------
\ProcessOptions
%--------------------------- package loading ----------------------------------
\LoadClass[12pt,a4paper]{report}
%-------------------------------- main code -----------------------------------
\renewcommand{\bibname}{References}

\RequirePackage{amssymb}
%\RequirePackage{graphics}
%\RequirePackage{graphicx}
%\RequirePackage{fancyhdr}
%\RequirePackage{eucal}
\RequirePackage[USenglish]{babel}
%\RequirePackage[usenames, dvipsnames]{color}
%\RequirePackage[perpage]{footmisc}
\RequirePackage[square, sort, numbers, authoryear]{natbib}
%\RequirePackage{ifthen}

\long\def\comment#1{}
\comment{
\usepackage[ pdftex,
            pdfpagelayout = None, %useoutlines,
            %bookmarks,
            %bookmarksopen = true,
            pdfstartview = FitH, %other options: Fit, FitB
            bookmarksnumbered = true,
            breaklinks = true,
            linktocpage,
            backref,
            colorlinks = true,
            linkcolor = blue,
            urlcolor  = blue,
            citecolor = blue,
            anchorcolor = blue,
            hyperindex = false,
            hyperfigures,
            a4paper,
            pdfpagemode = None,
            pdfauthor = {M. Devi Prasad},
            pdftitle = {Aspect \ Oriented \ Software \ Development},
            ]{hyperref}
}
% We need 1" margins except on the binding edge, where it is 1 1/2"

\oddsidemargin 0in \evensidemargin 0in
\marginparwidth 40pt \marginparsep 10pt
%\topmargin 0pt 
\headsep .5in
%\textheight 8.1in 
%\textwidth 6in

\textwidth 15cm
\textheight 21cm
%\footheight 0cm
%\headheight 0cm
\topmargin 0.5cm
\topskip 0.5cm


% Disallow page breaks at hyphens (this will give some underfull
% vbox's, so an alternative is to use \brokenpenalty=100 and manually 
% search for and fix such page breaks)

\brokenpenalty=10000

% Use 1.3 times the normal baseline-to-baseline skip
\renewcommand{\baselinestretch}{1.3}

% Figures and tables are to be numbered sequentially throughout the
% thesis, rather than within chapters; for style consistency, do
% equations the same.  The following \def of \cl@chapter is a kludge:
% really there should be a \removefromreset, to stop figure numbers,
% etc., from being reset at chapter beginnings
\def\cl@chapter{\@elt{section}\@elt{footnote}}
\def\thefigure{\@arabic\c@figure}
\def\thetable{\@arabic\c@table}
\def\theequation{\arabic{equation}}

% Redefine the macro used for floats (including figures and tables)
% so that single spacing is used. (Note \def\figure{\@float{figure}set
% single spacing} doesn't work because figure has an optional
% argument)
\def\@xfloat#1[#2]{\ifhmode \@bsphack\@floatpenalty -\@Mii\else
   \@floatpenalty-\@Miii\fi\def\@captype{#1}\ifinner
\@parmoderr\@floatpenalty\z@
\else\@next\@currbox\@freelist{\@tempcnta\csname ftype@#1\endcsname
\multiply\@tempcnta\@xxxii\advance\@tempcnta\sixt@@n \@tfor \@tempa
:=#2\do {\if\@tempa h\advance\@tempcnta \@ne\fi \if\@tempa
t\advance\@tempcnta \tw@\fi \if\@tempa b\advance\@tempcnta
4\relax\fi \if\@tempa p\advance\@tempcnta 8\relax\fi
}\global\count\@currbox\@tempcnta}\@fltovf\fi
\global\setbox\@currbox\vbox\bgroup
\def\baselinestretch{1}\@normalsize \boxmaxdepth\z@
\hsize\columnwidth \@parboxrestore}

% Redefine the macro used for footnotes to use single spacing
\long\def\@footnotetext#1{\insert\footins{\def\baselinestretch{1}\footnotesize
    \interlinepenalty\interfootnotelinepenalty 
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth \@parboxrestore
   \edef\@currentlabel{\csname p@footnote\endcsname\@thefnmark}\@makefntext
    {\rule{\z@}{\footnotesep}\ignorespaces
      #1\strut}}}

% \author, \title are defined in report; here are the rest of the
% front matter defining macros
\def\dept#1{\gdef\@dept{#1}}
\def\supervisor#1{\gdef\@supervisor{#1}}
\def\degree#1{\mox{\gdef\@degree{#1}}}
\def\submitdate#1{\gdef\@submitdate{#1}}
\def\copyrightyear#1{\gdef\@copyrightyear{#1}} % \author, \title in report
\def\specialization#1{\mbox{\gdef\@specialization{#1}}}
\def\@title{}
\def\@author{}
\def\@dept{Computer Science and Engineering}
\def\@supervisor{}
\def\@degree{Bachelor of Technology}
\def\@specialization{Computer Science}
\def\@college{~Motilal Nehru National Institute of Technology, Allahabad}
\def\@submitdate{\ifcase\the\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi,
  \space \number\the\year}

%\ifnum\month=12
%    \@tempcnta=\year \advance\@tempcnta by 1
%    \edef\@copyrightyear{\number\the\@tempcnta}
%\else
    \def\@copyrightyear{\number\the\year}
%\fi
%
\newif\ifcopyright \newif\iffigurespage \newif\iftablespage
\copyrightfalse


\def\titlep{%
    \thispagestyle{empty}%
    \renewcommand{\baselinestretch}{1}
    \null\vskip1in%
    \begin{center}
%       {\sl Title of the Thesis}\\
%       \vspace{1cm}
%        {\Large\uppercase\expandafter{\@title}}}
        \textbf{{\Large\expandafter{\@title}}}
    \end{center}
    \vfill
    \begin{center}
%        {\sl A Thesis Submitted}\\
        {A Report Submitted}\\
        in Partial Fulfillment of the Requirements\\
        for the Degree of\\
        \textbf{\@degree}\\
        in\\
        \textbf{\@specialization}\\
    \end{center}
    \vfill
    \begin{center}
%        \sl by\\
        by\\
        \textbf{\@author}
    \end{center}
    \vfill
    \begin{center}
%        {\sl to the}\\
        {to the}\\
        {\bf \uppercase\expandafter{\@dept} DEPARTMENT}\\
        {\large MOTILAL NEHRU NATIONAL INSTITUTE OF TECHNOLOGY\\ALLAHABAD}\\
        \bf\@submitdate
    \end{center}
    \vskip.5in\newpage}

\def\copyrightpage{%
    \null\vfill
    \begin{center}
        \Large\copyright\ Copyright \@copyrightyear\\
        by\\
        \@author\\
    \end{center}
    \vfill\newpage}

\def\undertakingpage{\newpage%
    \renewcommand{\baselinestretch}{1.3}
    \vspace*{\fill}
    \centerline{\LARGE\bf UNDERTAKING}
    \vspace{3cm}
    \begin{center}
    \begin{minipage}{4.5in}
    \parindent=0pt
    I declare that the work presented in this report titled
    ``{\sl\@title}'', submitted to the \mbox{\@dept} Department, \@college, for 
    the award of the {\bf\emph \@degree} \ degree in \mbox{\bf\emph \@specialization}, 
    is my original work. I have not plagiarized or submitted the same work for 
    the award of any other degree. In case this undertaking is found incorrect, I accept 
    that my degree may be unconditionally withdrawn.\par
    \vspace*{2cm}
    \@submitdate\\Allahabad\par
    \hbox to 4.5in {
    \hfill\vbox{\hsize 2in\vrule width 2in height0.4pt\vspace*{-4mm}\par\center{(\@author)}}}
    \vspace*{5cm}
    \end{minipage}
    \end{center}
}

\long\def\signature#1{%
\begin{center}
\begin{minipage}{4.5in}
\parindent=0pt
Certified   that  the   work   contained   in   the  report   titled
``{\sl\@title}'', by {\sl\@author}, has been carried out under  my
supervision and that this work has not been submitted  elsewhere for a
degree.
\par
\vspace{3cm}
\hbox to 4.5in{\@submitdate\hfill
    \vbox{\hsize 3in\vrule width 2in height0.4pt\par#1}\hss}
\end{minipage}
\end{center}
}

\def\signaturepage{%
    \renewcommand{\baselinestretch}{1.3}
    \vspace*{\fill}
    \centerline{\LARGE\bf CERTIFICATE}
    \vspace{3cm}
    \signature{
        (\@supervisor)\par
        \@dept \ Dept.\par
        M.N.N.I.T, Allahabad\\\\\par
    }
    \vspace*{\fill}
}

\def\beforepreface{
    \pagenumbering{roman}
    \pagestyle{plain}
    \titlep
    \ifcopyright\copyrightpage\fi
    \undertakingpage
    \signaturepage}

\def\prefacesection#1{%
    \chapter*{#1}
    \addcontentsline{toc}{chapter}{#1}}


% The \addvspace business below is to avoid the extra space that gets put
% in the tables of figures and tables between chapters (it makes no sense
% now that the numbering is continuous)
\def\afterpreface{\newpage
    \tableofcontents
    \newpage
    \iftablespage
        {\addvspace{10pt}
        \let\saveaddvspace=\addvspace
        \def\addvspace##1{}
        \listoftables
        \let\addvspace=\saveaddvspace}
        \newpage
    \fi
    \iffigurespage
        {\addvspace{10pt}
        \let\saveaddvspace=\addvspace
        \def\addvspace##1{}
        \listoffigures
        \let\addvspace=\saveaddvspace}
        \newpage
    \fi
    \pagenumbering{arabic}
    \pagestyle{plain}
}

\setcounter{secnumdepth}{3}

\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{-3.25ex plus  -1ex minus -.2ex}{1.5ex plus .2ex}{\large\sl}}
\def\thesubsubsection{\hbox{\vrule height 1ex width .75ex depth .3ex}}

%
%
% Some Theorem Environments
%
%


\newtheorem{condition}{Condition}
\newtheorem{proposition}{Proposition}[chapter]
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{defn}{Definition}[chapter]
\newtheorem{conjecture}{Conjecture}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{problem}{Problem}[chapter]
\newtheorem{subproblem}{Subproblem}[chapter]


\newenvironment{definition}{\begin{defn}\em}{\end{defn}}
\newenvironment{proof}{\noindent{\bf Proof:}\hspace  {5mm}}{
\nopagebreak\nobreak\quad\nopagebreak\nobreak\proofbox\medskip}
\newenvironment{example}{\noindent{\bf Example}\quad}{
\nopagebreak\nobreak\quad\nobreak\nopagebreak\proofbox\smallskip}

\newcounter{case}
\newenvironment{case}{\begin{list}{\bf Case \arabic{case}:}%
{\usecounter{case}%
\labelwidth 5em
\leftmargin 5em
\setlength\listparindent{\parindent}
\makelabel}
\def\case{\item }}{\end{list}\nobreak\nopagebreak\quad\nopagebreak\nobreak\proofbox\bigskip}
\def\proofbox{\hbox{\vrule height1.2ex depth.3ex width0.8ex}}


% Get References instead of Bibliography
\def\thebibliography#1{\chapter*{References\@mkboth
% {BIBLIOGRAPHY}{BIBLIOGRAPHY}}
 {References}{References}}
 \addcontentsline{toc}{chapter}{References}%\bf
 \list
 {[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
 \advance\leftmargin\labelsep
 \usecounter{enumi}}
 \def\newblock{\hskip .11em plus .33em minus .07em}
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist

% --> Remove dots in TOC [24/10/2001] Thanks to the tocloft documentation! [10/11/2001]
% \renewcommand{\@dotsep}{10000}
