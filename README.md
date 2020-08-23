# Reconciling Kinetic and Equilibrium Models of Bacterial Transcription
Welcome to the GitHub repository for the bursty transcription project! This
repository serves as a record for work described in the publication "Reconciling
Kinetic and Equilibrium Models of Bacterial Transcription"

## Installation
The intend of this repository is to make every step of the publication
completely transparent and reproducible. The project involved a significant
amount of home-grown Python code that we wrapped as a module `srep`. To
install the package first you need to make sure that you have all of the
required dependencies. To check for this you can use 
[`pip`](pypi.org/project/pip) by executing the following command:

``` pip install -r requirements.txt ```

Once you have all of the packages installed locally, you can install our custom
module by running the following command:

``` pip install -e ./ ```

When installed, a new folder `srep.egg-info` will be installed. This folder
is required for the executing of the code in this repository.

## Repository Architecture
For convenience the repository is divided into several directories and
subdirectories. Please see each directory for information regarding each file.

### **`code`**
This folder contains all the source code used for the project. From the
parameter inference done on experimental data, to the generation of all figures.
The directory is broken up into the following subdirectories:
1. **`analysis`**: This folder contains the final analysis done over the
   single-molecule mRNA FISH data to infer the kinetic parameters.
2. **`exploratory`**: This folder contains folder that did not make it to the
   final analysis. In oder to be transparent and leave a record of all the
   things attempted in this work we include every piece of code generated.
3. **`figures`**: This folder contains individual scripts to generate all plots in
   the manuscript. The folder is further divided into two folders `main` and
   `si` that each contain the final versions of each of the figures presented
   both in the main text and in the supplementary material, respectively.
4. **`stan`**: This folder contains all of the code generated for the statistical
   software `Stan`. This software implements the state-of-the-art Hamiltonian
   Monte Carlo sampler that we used in part of our parameter inference over the
   experimental data.

### **`data`**
This directory contains the experimental data used in this manuscript.
Specifically we make use of the single-molecule mRNA FISH first reported by
[Jones et al. 2014](https://science.sciencemag.org/content/346/6216/1533). See
the `README` file in this directory for further details.

### **`figures`**
This folder contains the PDF version of all figures found both in the main text
and in the supplemental material.

### **`srep`**
This directory contains all the files necessary for the installation of the
`srep` module utilized in this work. See `README` file inside directory for
further details.

### **`tests`**
This folder contains the software tests necessary to check the proper
installation of the required packages. Specifically it contains a test for the
proper installation of `Stan`, the statistical software used for part of the
parameter inferences done in this project.

## License
![](https://licensebuttons.net/l/by/3.0/88x31.png)

All creative works (writing, figures, etc) are licensed under the [Creative
Commons CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. All
software is distributed under the standard MIT license as follows

```
Copyright 2020 The Authors 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```