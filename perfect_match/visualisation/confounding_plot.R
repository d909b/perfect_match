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

# This file plots the results of the TCGA hidden confounding experiment (Fig. 4 in the paper).
library(latex2exp)

makeTransparent = function(..., alpha=0.15) {
  # From: https://stackoverflow.com/a/20796068
  if(alpha<0 | alpha>1) stop("alpha must be between 0 and 1")

  alpha = floor(255*alpha)
  newColor = col2rgb(col=unlist(list(...)), alpha=FALSE)
  .makeTransparent = function(col, alpha) {
    rgb(red=col[1], green=col[2], blue=col[3], alpha=alpha, maxColorValue=255)
  }
  newColor = apply(newColor, 2, .makeTransparent, alpha=alpha)
  return(newColor)
}

# The following numbers were extracted from the finished experiments using run_results.sh.
pbm_mse_pehe <- c(19.57, 16.21, 14.34, 11.79, 11.05, 11.36, 10.25, 11.29, 10.48)
pbm_mse_pehe_std <- c(4.02, 2.05, 3.31, 2.40, 0.87, 1.68, 0.81, 2.08, 0.47)
pbm_mse_ate <- c(7.83, 5.73, 4.82, 3.47, 2.60, 2.93, 2.39, 2.96, 2.67)
pbm_mse_ate_std <- c(2.23, 1.10, 1.82, 1.26, 0.51, 0.99, 0.36, 0.92, 0.27)

tarnet_mse_pehe <- c(17.52, 14.83, 15.61, 14.72, 12.19, 11.12, 11.35, 10.65, 10.50)
tarnet_mse_pehe_std <- c(3.11, 2.35, 2.98, 2.81, 2.19, 1.64, 1.70, 1.52, 1.49)
tarnet_mse_ate <- c(6.33, 4.89, 5.34, 4.94, 3.40, 2.93, 3.19, 2.93, 2.96)
tarnet_mse_ate_std <- c(2.02, 1.46, 1.71, 1.58, 0.99, 0.59, 0.74, 0.94, 0.68)

cfrnet_mse_pehe <- c(19.48, 14.02, 14.13, 14.25, 11.81, 11.81, 11.40, 12.02, 11.70)
cfrnet_mse_pehe_std <- c(3.76, 2.12, 2.46, 2.36, 1.17, 0.82, 1.07, 1.05, 0.96)
cfrnet_mse_ate <- c(7.67, 4.01, 4.58, 4.99, 3.01, 2.92, 2.83, 3.07, 3.07)
cfrnet_mse_ate_std <- c(2.52, 1.25, 1.25, 1.92, 0.84, 0.29, 0.67, 0.82, 0.77)

tarnetpd_mse_pehe <- c(19.56, 16.74, 16.74, 15.84, 15.10, 14.37, 14.23, 13.91, 13.68)
tarnetpd_mse_pehe_std <- c(3.89, 3.72, 3.86, 3.45, 2.86, 2.81, 2.65, 2.39, 2.55)
tarnetpd_mse_ate <- c(7.04, 7.70, 8.36, 8.32, 7.50, 7.42, 7.19, 7.17, 7.07)
tarnetpd_mse_ate_std <- c(2.83, 2.47, 2.66, 2.72, 2.00, 2.21, 1.71, 1.79, 1.99)

kappas <- c(10, 20, 30, 40, 50, 60, 70, 80, 90)
pehe_min <- 9
pehe_max <- 23
pehe_title <- TeX("$\\sqrt{\\epsilon_{PEHE}}$")
pehe_title_cex <- 3

ate_min <- 1.5
ate_max <- 11
ate_title <- TeX("$\\epsilon_{ATE}$")
ate_title_cex <- 3

for(metric in c("pehe", "ate")) {
  # Plot setup.
  pdf(paste("confounding_", metric, ".pdf", sep=""), width=14)
  par(mar = c(5.1, 5.6, 4.1, 2.1))

  # Get per-metric config.
  min_y <- get(paste(metric, "_min", sep=""))
  max_y <- get(paste(metric, "_max", sep=""))
  title <- get(paste(metric, "_title", sep=""))
  title_cex <- get(paste(metric, "_title_cex", sep=""))

  plot(kappas, tarnet_mse_pehe, type = 'n', xlim=c(min(kappas), max(kappas)), ylim=c(min_y, max_y),
       cex.axis=2, ann=FALSE, mgp=c(3, 1, 0), xaxs = "i", yaxs = "i")
  mtext(side=1, text=substitute(paste("Percentage of Hidden Confounding [%]")), line=4, cex=3)
  mtext(side=2, text=title, line=2.75, cex=title_cex)

  # Plot each method.
  i <- 1
  postfix_mean <- paste("_", metric, sep="")
  postfix_std <- paste("_", metric, "_std", sep="")
  colors <- c('#F59799', '#9DC7EA', '#FDC67C', '#A75CC6', '#A7916D', '#666666')
  methods <- c("pbm_mse", "cfrnet_mse", "tarnet_mse", "tarnetpd_mse")
  names <- c("PM", "CFRNET", "TARNET", "PD")
  for(method in methods) {
    mean <- rev(get(paste(method, postfix_mean, sep="")))
    std <- rev(get(paste(method, postfix_std, sep="")))
    high <- mean + std
    low <- mean - std

    polygon(c(kappas, rev(kappas)), c(high, rev(low)),
            col=makeTransparent(colors[i]), border = NA)

    lines(kappas, mean, lwd=5, cex=3, pch=i-1, type="o", col=colors[i])
    lines(kappas, high, lwd=1, lty="dashed", col=colors[i])
    lines(kappas, low, lwd=1, lty="dashed", col=colors[i])

    i <- i + 1
  }
  legend("topleft", legend=names, pch=0:length(methods), col=colors, cex=2, lwd=5)
  dev.off()
}