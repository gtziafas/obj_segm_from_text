import subprocess

for idx in range(41):
	if not int(.1*idx):
		idx = "0" + str(idx) + "/"
	else:
		idx = str(idx) + "/"

	path = "/home/s3913171/zsgnet-pytorch/data/referit/saiapr_tc_12/" + idx
	bash = "find " + path + " -type f -print0 | xargs -0 mv -t /home/s3913171/zsgnet-pytorch/data/referit/saiapr_tc_12/"

	process = subprocess.Popen(bash.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

