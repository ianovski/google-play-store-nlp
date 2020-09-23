from time import gmtime, strftime
import subprocess

# Build a Spark Docker Image to Run the Processing Job
build_command = "sudo docker build -t spark-image -f container/Dockerfile container"
check_build_command = "sudo docker inspect spark-image"
# build_command = "pwd"
process = subprocess.Popen(build_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print("[debug] building spark-image..... \n[debug] output = {}\n[debug] error = {}".format(output,error))

process = subprocess.Popen(check_build_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print("[debug] checking docker image..... \n[debug] output = {}\n[debug] error = {}".format(output,error))

