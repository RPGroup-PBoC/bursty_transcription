## `data`

This folder contains tidy dataframes with the experimental data used for the 
parameter inference in the main text. Specifically we have
- **`brewster_jones_2012.csv`**: List of different RNA polymerase binding site
  energies as inferred in [Jones, Brewster, and Phillips,
  2012](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002811).
  These energies were mapped from an energy matrix that assumes linear
  contributions of each of the base pairs to the total binding energy of the
  polymerase, mapping DNA sequences to binding energies in kʙT units.
- **`jones_brewster_2014.csv`**: Tidy dataframe containing single molecule mRNA
  counts obtained for a series of **unregulated promoters** as reported in
  [Jones et al. 2014](https://science.sciencemag.org/content/346/6216/1533).
- **`jones_brewster_regulated_2014.csv`**: Tidy dataframe containing single
  molecule mRNA counts obtained for a series of **regulated promoters** as
  reported in [Jones et al.
  2014](https://science.sciencemag.org/content/346/6216/1533). 