# ROS-Neuro qda decoder package

This package implements a QDA classifier as a plugin for rosneuro::decoder::Decoder and as class. The test used it as class, for the usage as rosnode it misses the tensor message after the pwelch computation.

## Usage
The package required as ros parameter:
<ul>
    <li> <b>cfg_name</b>: which is the name of the structure in the yaml file; </li>
    <li> <b>yaml file</b>: which contains the structure for the qda classifier. </li>
</ul>

## Example of yaml file
```
QdaCfg:
  name: "qda"
  params:
    filename: "file1"
    subject: "s1"
    n_classes: 2
    class_lbs: [771, 773]
    n_features: 2
    idchans: [1, 2]
    freqs: "10;
            20;"
    priors: [0.5, 0.5]
    lambda: 0.5
    means: "0.4948 3.4821;
            0.4647 3.4921;" 
    covs: " 0.9273  0.9325;
           -0.0187 -0.0144;
           -0.0187 -0.0144;
            0.9120  0.8959;"
```

Some parameters are hard coded:
<ul>
    <li> <b>idchans</b>: the index of the channels from 1 to the number of channels used; </li>
    <li> <b>freqs</b>: the selected frequencies; </li>
    <li> <b>means</b>: matrix of [features x classes]; </li>
    <li> <b>covs</b>: matrix of [(feature * feature) x classes]. Therefore, each column is reshaped to obtain the covariance matrix of size [feature x feature] for that class. Thus the the vector paased must be obtained by concatenate columns. </li>
</ul>