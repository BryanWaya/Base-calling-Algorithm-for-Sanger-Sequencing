# Base Calling Algorithm for Sanger Sequencing

Overview

This project presents a model-based signal processing and base-calling algorithm for first-generation Sanger DNA sequencing. The system processes raw electropherogram signals, performs noise reduction, baseline correction, channel separation, peak detection, and sequence reconstruction to generate accurate DNA sequences from .SRD chromatogram files.

The project was developed as part of a Master's Thesis in Bioinformatics and Computational Biology.

---

## Features

* Baseline correction of fluorescence signals
* Multi-channel dye signal processing
* Peak detection and alignment
* Signal-to-noise ratio filtering
* Base calling using chromatogram peak analysis
* NCBI BLAST integration for sequence validation
* Graphical User Interface (GUI) for ease of use

---

## Project Structure

```text
Thesis work/
├── ud_GUI.py          # Main GUI application
├── ud_processor.py    # Signal processing and base calling
└── blast_ncbi.py      # NCBI BLAST validation
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/BryanWaya/Base-calling-Algorithm-for-Sanger-Sequencing.git
cd Base-calling-Algorithm-for-Sanger-Sequencing
```

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the GUI:

```bash
python "Thesis work/ud_GUI.py"
```

Load a chromatogram (.srd) file and process the sequence.

---

## Sample Input

Input:

```text
sample_data/example.srd
```

Output:

```text
ATGCGTACCGTAGGCTTACGATCGATCG
```

---

## Methodology

The algorithm performs the following stages:

1. Raw signal extraction
2. Baseline correction
3. Noise suppression
4. Signal normalisation
5. Peak detection
6. Channel discrimination
7. Base calling
8. Sequence generation
9. BLAST verification

---

## Performance Evaluation

The proposed algorithm was evaluated using Sanger sequencing chromatograms and compared against conventional base-calling approaches.

Performance metrics include:

* Sequence identity (%)
* Read length
* Signal-to-noise ratio
* Number of ambiguous calls
* BLAST alignment score

Results demonstrated improved sequence reconstruction in noisy chromatograms and longer reliable read lengths compared with traditional peak-based methods.

---

## Author

  Iruoghene Bryan Waya

Master's Thesis: 

Model-Based Signal Processing and Base Calling for First-Generation Sanger Sequencing
