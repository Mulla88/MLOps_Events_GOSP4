pipeline {
    agent any
    stages {
        stage('Clone') {
            steps {
                git 'https://github.com/your-repo.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t my_rnn_model .'
            }
        }
        stage('Run Tests') {
            steps {
                sh 'docker run my_rnn_model pytest tests/'
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                sh 'kubectl apply -f deployment.yaml'
            }
        }
    }
}
