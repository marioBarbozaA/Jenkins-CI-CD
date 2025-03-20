pipeline {
    agent any  // Usa cualquier máquina disponible en Jenkins
    tools {
        python 'MiPython3.9' 
    }
    stages {
        stage('Clonar Repositorio') {
            steps {
                git 'https://github.com/marioBarbozaA/Jenkins-CI-CD.git'
            }
        }
        stage('Instalar Dependencias') {
            steps {
                bat  'pip install -r requirements.txt'
            }
        }
        stage('Verificar Python') {
            steps {
                bat 'python --version'
            }
        }
        stage('Entrenar Modelo') {
            steps {
                bat  'python main.py train'
            }
        }
        stage('Desplegar API') {
            steps {
                bat  'uvicorn main:app --host 0.0.0.0 --port 8000 &'
            }
        }
    }
}