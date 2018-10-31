# Copyright (C) 2018  Patrick Schwab, ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions
#  of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
generateIHDP <- function(iter, output_dir, setting = "A") {
	covariates <- "select"
	p.score <- "none"
	verbose <- FALSE
	require(npci, quietly=TRUE)

	setwd("/home/d909b/thirdparty/npci/examples/ihdp_sim/")
	source("/home/d909b/thirdparty/npci/examples/ihdp_sim/results.R")

	opt <- list()
	L <- list()
	x.0 <- list()
	opt.0 <- list()
	L.0 <- list()
	x.0.0 <- list()
	opt.1 <- list()
	L.1 <- list()
	x.0.1 <- list()

	source("/home/d909b/thirdparty/npci/examples/ihdp_sim/data.R")
	loadDataInCurrentEnvironment(covariates, p.score)

	w <- 0.5
	overlap <- FALSE
	generateDataForIterInCurrentEnvironment(iter, x, z, w, overlap, covariates, setting)

	write.csv(x, file = file.path(output_dir, "x.csv"))
	write.csv(y, file = file.path(output_dir, "y.csv"))
    write.csv(y.0, file = file.path(output_dir, "y.0.csv"))
    write.csv(y.1, file = file.path(output_dir, "y.1.csv"))
    write.csv(mu.0, file = file.path(output_dir, "mu.0.csv"))
    write.csv(mu.1, file = file.path(output_dir, "mu.1.csv"))
	write.csv(z, file = file.path(output_dir, "z.csv"))
}