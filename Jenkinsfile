pipeline {
    agent any

    environment {
        PYTHON = 'python3'
        MODEL = 'RF'
    }

    stages {
        stage('Run ML Pipeline in Container') {
            steps {
                script {
                    sh '''
                    docker run --rm \
                        -e PYTHON=${PYTHON} \
                        -e MODEL=${MODEL} \
                        -v $WORKSPACE:/app \
                        youva1/my-ml-app python3 run_pipeline.py
                    '''
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
