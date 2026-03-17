# Reconnaissance Faciale avec OpenCV (Android IP Webcam)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision%20Par%20Ordinateur-red.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Calcul%20Scientifique-orange.svg)](https://numpy.org/)
[![Licence: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Un **système complet de reconnaissance faciale construit de zéro avec OpenCV**, utilisant un **téléphone Android comme webcam IP**.  
Ce projet montre tout le **pipeline de vision par ordinateur** : création du dataset, entraînement du modèle et reconnaissance en temps réel.

---

## 🎯 Vue d'ensemble

Pipeline de reconnaissance faciale en **temps réel** utilisant l’algorithme **LBPH (Local Binary Patterns Histogram)**.

| Étape                  | Description                                  |
|------------------------|----------------------------------------------|
| **Création du dataset** | Capture d’images faciales via webcam Android |
| **Entraînement**        | Entraîner le modèle LBPH avec les images    |
| **Reconnaissance temps réel** | Détection et identification faciale sur flux vidéo |

Pipeline :  
Camera Stream → Détection faciale → Extraction de caractéristiques → Classifieur LBPH → Prédiction identité


---

## 💼 Importance

| Compétence démontrée        | Valeur métier                                          |
|-----------------------------|--------------------------------------------------------|
| Ingénierie vision par ordinateur | Création complète de pipelines avec OpenCV          |
| Compréhension algorithme     | Utilisation de LBPH au lieu d’APIs “boîte noire”      |
| Gestion du dataset           | Collecte, prétraitement et entraînement du modèle    |
| Temps réel                   | Traitement efficace du flux vidéo                     |
| Edge AI                      | Algorithmes légers adaptés aux appareils embarqués    |

---

## 🚀 Techniques principales

| Composant          | Technologie                 |
|-------------------|-----------------------------|
| Détection faciale  | Haar Cascade Classifier     |
| Extraction         | LBP (Local Binary Pattern)  |
| Reconnaissance     | Classifieur LBPH           |
| Source vidéo       | Android IP Webcam          |
| Traitement image   | OpenCV + NumPy             |

---

## 🧠 Algorithme

LBPH (Histogramme de Patterns Binaires Locaux) :
LBP(x, y) = Σ s(gi - gc) * 2^i

| Symbole | Signification     |
|---------|-----------------|
| `gc`    | Pixel central    |
| `gi`    | Pixel voisin     |
| `s(x)`  | Fonction seuil   |

---

## 📂 Structure du projet

| Fichier                 | But                                   |
|-------------------------|--------------------------------------|
| face_dataset.py          | Collecter images depuis la caméra    |
| training.py              | Entraîner le modèle LBPH             |
| face_recognition.py      | Reconnaissance en temps réel         |
| dataset/                 | Images pour entraînement             |
| trainer/                 | Modèle entraîné                      |

---

## ⚙️ Workflow

Démarrer IP Webcam
↓
Collecte dataset
↓
Entraînement modèle
↓
Reconnaissance faciale temps réel


---

## 📸 Collecte du dataset

python face_dataset.py

- Connecte la webcam IP du téléphone  
- Détecte les visages  
- Capture **30 échantillons par utilisateur**  
- Enregistre dans `dataset/`

Format des fichiers :
User.<ID>.<NumSample>.jpg

---

## 🧠 Entraînement du modèle

python training.py

- Charge les images de `dataset/`  
- Extrait les IDs  
- Détecte les visages  
- Entraîne un **reconnaisseur LBPH**  
- Sauvegarde le modèle : `trainer/trainer.yml`

---

## 🎥 Reconnaissance temps réel

python face_recognition.py

- Démarre le flux caméra  
- Détecte les visages  
- Prédit l’identité  
- Affiche les résultats

Exemple :

User1 89%
User2 92%
Unknown

---

## 📦 Installation

git clone https://github.com/N1N0u/facerecon.git

cd facerecon
pip install opencv-python numpy pillow

Télécharger Haar Cascade :  
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

---

## 📱 Configuration caméra Android

- Installer **IP Webcam**  
- Démarrer le serveur et obtenir URL :
http://PHONE_IP:8080/video
- Mettre à jour le script :
cv2.VideoCapture("http://PHONE_IP:8080/video")


---

## 🧪 Cas d’usage

| Application        | Description                        |
|-------------------|------------------------------------|
| Accès porte smart  | Identifier utilisateurs autorisés  |
| Systèmes présence  | Détecter employés enregistrés      |
| Caméras sécurité   | Surveiller individus connus        |
| Expériences AI     | Apprendre la vision par ordinateur |
| Maison intelligente| Reconnaître membres de la famille  |

---

## 🔬 Apprentissage

- Vision par ordinateur  
- Traitement d’image  
- Extraction de caractéristiques  
- Machine Learning (LBPH)  
- Vidéo temps réel  
- Architecture OpenCV  

---

## 🚧 Améliorations futures

- Reconnaissance faciale deep learning (FaceNet / ArcFace)  
- Accélération GPU  
- Multi-caméras  
- API REST pour reconnaissance à distance  
- Dashboard web  
- Gestion base de visages  

---

## 📝 Licence

MIT — usage éducatif et commercial libre

---

## 👨‍💻 Auteur

**N1N0u**

