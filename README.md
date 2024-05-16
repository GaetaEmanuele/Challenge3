# Challenge3 #
this is the README file for the 3rd challege of PACS corse
in the src folder there are the needed file to solve the problem of the Laplacian with omogeneous boundary condition. The subfolder test contins a Jason file that should be modified by the user for adapth th problem in his/her context, There is also another subfolder containing the result of the problem already solved

# How to install the code 
on terminal you have to digit: git clone git@github.com:GaetaEmanuele/Challenge3.git

# How run the code
In the folder src there is a MakeFile that use a the macro PACS_ROOT.
1. In case you do not have that variables before running the code on terminale you must digit export PACS_ROOT=complete_path_to_pacs-exampples_folder/Examples. for having the full path it is enought to go in our local repository pacs-examples/Examples and on terminal digits pwd
2. In case your device is a MAC book and you use docker for having the linux machine you also must set the variable LD_LIBRARY_PATH. On the terminal you have to write export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/full-path/to/the/library in our case will be the sub folder lib of pacs-examples/Examples
3. Now in Challenge3/src you can do make on terminal 

