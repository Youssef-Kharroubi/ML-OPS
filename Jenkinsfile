pipeline {
    agent {
        docker {
            image 'youva1/my-ml-app'  
            args '-v $WORKSPACE:/app -w /app'         
        }
    }

    environment {
        PYTHON = 'python3'
        MODEL = 'default_model'
    }

    stages {
        stage('Train Model') {
            steps {
                script {
                    sh '${PYTHON} main.py train --train_data churn-bigml-80.csv --model ${MODEL} --save_model models/${MODEL}.pkl'
                }
            }
        }

        stage('Test Model') {
            steps {
                script {
                    sh '${PYTHON} main.py test --test_data churn-bigml-20.csv --load_model models/${MODEL}.pkl'
                }
            }
        }

        stage('Run Model') {
            steps {
                script {
                    sh '${PYTHON} main.py --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --model ${MODEL}'
                }
            }
        }

        stage('Cleanup') {
            steps {
                script {
                    sh 'rm -rf __pycache__ logs/*.log'
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed! Check the logs for errors.'
        }
    }
}
