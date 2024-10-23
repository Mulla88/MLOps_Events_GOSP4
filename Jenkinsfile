pipeline {
    agent any
    stages {
        stage('Clone Repository') {
            steps {
                // Pull the repository from the correct branch (replace 'main' with your branch if necessary)
                git branch: 'main', url: 'https://github.com/Mulla88/MLOps_Events_GOSP4'
            }
        }
        
        stage('Check Docker') {
            steps {
                // Check if Docker is installed and running
                script {
                    def dockerCheck = sh(script: 'docker --version', returnStatus: true)
                    if (dockerCheck != 0) {
                        error("Docker is not installed or accessible.")
                    }
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                // Build the Docker image
                sh 'docker build -t my_rnn_model .'
            }
        }
        
        stage('Run Tests') {
            steps {
                // Run tests inside the Docker container
                sh 'docker run my_rnn_model pytest tests/'
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                // Deploy to Kubernetes
                sh 'kubectl apply -f deployment.yaml'
            }
        }
    }
}
