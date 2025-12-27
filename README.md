# Nonai
A python demo script that allow detecting generated media content.
Currently this script only detects generated AI audio files ( it supports .wav files currently ).
## How to use :
Download the ED-script.py, Detect-script.py and the requierments.txt
Run : 
```bash
pip install -r requirements.txt
```
Create a folder called data with this following structure :
data-->ai-->( place here your AI generated audio files .wav for the training )
    -->humain-->(place here your real audio files .wav for the training )
Run the ED-script.py to train the Random Forset model.
Run the Detect-script.py to run the detection and edit this var "nom_fichier =" to select the .wav file to be checked.
## Important: 
This is a demo project that demonstrate how to detect generated AI voices, IT ISN'T BULLET PROOF OR READY TO BE DEPLOYED ON A LARGE SCALE ALSO IT MIGHT BE BYPASSED EASILY!
