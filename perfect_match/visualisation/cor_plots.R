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

# See README.md for how to generate nn_pehe.txt, pehe.txt and mse.txt from the experiment logs.
nn_pehe <- as.numeric(readLines(file("nn_pehe.txt")))
pehe <- as.numeric(readLines(file("pehe.txt")))
mse <- as.numeric(readLines(file("mse.txt")))

set.seed(909)
samples <- sample.int(length(pehe), 1400)
cor_method <- "pearson"

mse_p <- mse[samples]
pehe_p <- pehe[samples]
nn_pehe_p <- nn_pehe[samples]

pdf("mse.pdf")
plot(mse_p[mse_p < 1000 & pehe_p < 100], pehe_p[mse_p < 1000 & pehe_p < 100], pch=20, cex=1.5,
     ann=FALSE, mgp=c(3, 0.5, 0))
mtext(side = 1, text = "MSE", line = 2, cex=2)
mtext(side = 2, text = "PEHE", line = 2, cex=2)
op <- par(cex = 3)
correlation_coefficient_mse <- format(round(cor(mse, pehe, method=cor_method), 2), nsmall = 2)
legend("bottomright", legend=substitute(paste(rho, "=", corr), list(corr=correlation_coefficient_mse)), bty="n")
abline(lm(mse ~ pehe))
dev.off()

pdf("nn_pehe.pdf")
plot(nn_pehe_p[pehe_p < 100], pehe_p[pehe_p < 100], pch=20, cex=1.5,
     ann=FALSE, mgp=c(3, 0.5, 0))
mtext(side = 1, text = "NN-PEHE", line = 2, cex=2)
mtext(side = 2, text = "PEHE", line = 2, cex=2)
op <- par(cex = 3)
correlation_coefficient <- format(round(cor(nn_pehe, pehe, method=cor_method), 2), nsmall = 2)
legend("bottomright", legend=substitute(paste(rho, "=", corr), list(corr=correlation_coefficient)), bty="n")
abline(lm(pehe ~ nn_pehe))
dev.off()
