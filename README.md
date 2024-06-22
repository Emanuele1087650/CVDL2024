# CVDL2024

## Overview

Questo progetto implementa un'architettura a due step basato su YOLOv10 e YOLOv8 per la detection e la classificazione di pescatori in immagini. Utilizza di default Grad-CAM per la detection e EigenCAM per la classificazione per visualizzare dove la rete si è maggiormente concentrata per prendere la sua decisione.

## Installazione

È consigliato utilizzare `Conda` per gestire l'ambiente virtuale e le dipendenze del progetto.

Creare e attivare un ambiente `Conda`:

```bash
conda create -n net2step python=3.9
conda activate net2step
pip install -r requirements.txt
```
## Utilizzo

La funzione main accetta tre parametri opzionali per eseguire la detection e la classificazione a due step.

+ `<immagine>`: Percorso dell'immagine su cui eseguire la detection e la classificazione.
+ `<Grad-Cam>`: Specifica se utilizzare Grad-Cam per la detection. Valori accettati: `True` o `False`.
+ `<EigenCAM>`: Specifica se utilizzare EigenCAM per la classificazione. Valori accettati: `True` o `False`.

Un esempio di utilizzo è il seguente:

```bash
python3 net2step.py '<immagine>' <cam_detection> <cam_classification>
```

Se non si desidera provare su una singola immagine, nella cartella di progetto sono presenti dele immagini di prova.
Con il comando seguente verrà fatta l'inferenza su tutte le immagini in cartella.
```bash
python3 net2step.py
```

## Risultati

I risultati verranno salvati all'interno della cartella di progetto, in particolare nella cartella `Risultati`.
Ogni immagine processata avrà una propria sotto-cartella con il nome dell'immagine.

```css
Risultati/
└── NomeImmagine/
    ├── classificazione/ (opzionale)
        ├── eigen_cam0.jpg
    ├── cam_detection.jpg (opzionale)
    └── detection.jpg
```

