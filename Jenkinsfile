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
                    // Check if Docker is accessible and available
                    def dockerCheck = sh(script: 'docker --version', returnStatus: true)
                    if (dockerCheck != 0) {
                        error("Docker is not installed or accessible.")
                    }
                }
            }
        }
        
        stage('Force Docker Permissions') {
            steps {
                script {
                    // Attempt to force Docker permissions (requires sudo or root access)
                    try {
                        // Add the jenkins user to the docker group
                        sh 'sudo usermod -aG docker jenkins || true'
                        
                        // Ensure correct group permissions for the Docker socket
                        sh 'sudo chown root:docker /var/run/docker.sock || true'
                        
                        // Restart Jenkins if necessary
                        sh 'sudo service jenkins restart || true'
                    } catch (Exception e) {
                        echo "Failed to adjust Docker permissions, continuing to attempt Docker usage."
                    }
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                // Build the Docker image, will fail if permissions aren't right
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
