pipeline {
    agent any  

    environment {
        PYTHON = 'python3'
        MODEL = 'RF'
        MPLCONFIGDIR = '/tmp'
        FONTCONFIG_PATH = '/tmp/.fontconfig' 
    }

    stages {
        stage('Verify Docker') {
            steps {
                script {
                    sh 'docker --version' 
                }
            }
        }
        
        stage('Pull Docker Image') {
            steps {
                script {
                    def myApp = docker.image('youva1/my-ml-app')
                    myApp.pull() // Pull latest image
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    def myApp = docker.image('youva1/my-ml-app')
                    myApp.inside('-v $WORKSPACE:/app -w /app -e MPLCONFIGDIR=$MPLCONFIGDIR -e FONTCONFIG_PATH=$FONTCONFIG_PATH') {
                        sh '${PYTHON} main.py train --train_data churn-bigml-80.csv --model ${MODEL} --save_model models/${MODEL}.pkl'
                    }
                }
            }
        }

        stage('Test Model') {
            steps {
                script {
                    def myApp = docker.image('youva1/my-ml-app')
                    myApp.inside('-v $WORKSPACE:/app -w /app -e MPLCONFIGDIR=$MPLCONFIGDIR -e FONTCONFIG_PATH=$FONTCONFIG_PATH') {
                        sh '${PYTHON} main.py test --test_data churn-bigml-20.csv --load_model models/${MODEL}.pkl'
                    }
                }
            }
        }

        stage('Run Model') {
            steps {
                script {
                    def myApp = docker.image('youva1/my-ml-app')
                    myApp.inside('-v $WORKSPACE:/app -w /app -e MPLCONFIGDIR=$MPLCONFIGDIR -e FONTCONFIG_PATH=$FONTCONFIG_PATH') {
                        sh '${PYTHON} main.py --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --model ${MODEL}'
                    }
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
