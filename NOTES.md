# NOTES



# Presentation


## Auto-Augmentation
-> Automatisierte Datenaugmentation
Eine Policy besteht aus (5) Sub-policies.
Eine Sub-Policy besteht aus 2 Operationen (z Bsp: Schere, Transformation, Spiegeln, Farbveränderung) mit jeweils einer Ausführungsintensität (10 Werte) und -wahrscheinlichkeit (11 Werte).

Vorteile:
    - Mindestens State-of-the-art Performance
    - Transferierbar zwischen unterschiedlichen Datensets

Alle Operationen (16):
ShearX/Y,TranslateX/Y, Rotate, AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Color, Brightness, Sharpness, Cutout, Sample Pairing

Für eine Sub-policy: (16x10x11)^2 Möglichkeiten.
5 zu wählen: 2,9e32



- hält U für R, 62 mal
- hält N für M, 48 mal
- hält H für G, 41 mal
- hält G für T, 38 mal
- hält U für D, 36 mal
- hält V für W, 26 mal



## Grad-CAM Analyse
- Allgemeine Beispiele
    - Aufbau mit Layers erklären
    - Oft vorkommende Phänomene beschreiben
        - Entscheidung an Bildgrenzen
        - "Drumrum"-Erkennen
        - zweiter Layer irrelevant?

- Surest aufzeigen
- Unsurest aufzeigen
Vielleicht das ganze noch kreuzen mit richtig/falsch?


## roadmap
- Berechnung der Surest/Unsurest
    - Codemäßig fertig machen
    - Jupyter Notebook fertig beschreiben
    - Präsentation Grad-CAM machen

(Format von Output von powerpoint zu jupyter notebook zurückändern)
