pipeline {
    agent any
    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/Mulla88/MLOps_Events_GOSP4'
            }
        }
        
        stage('Check Docker') {
            steps {
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
                // Build the Docker image within the Minikube context
                sh 'docker build -t my_rnn_model .'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                // Deploy the application to Minikube without validation
                sh 'kubectl apply -f deployment.yaml --validate=false'
            }
        }
    }
}
