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

# This file plots the results of the News-8 percentage of matched samples per batch experiment (Fig. 2 in the paper).

library(latex2exp)
makeTransparent = function(..., alpha=0.1) {
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

pdf("pm_percentage.pdf", width=14)

percentages <- 0:10*10

# The following numbers were extracted from the finished experiments using run_results.sh.
pehe <- c(22.39, 21.48, 20.87, 21.12, 20.89, 20.66, 20.78, 20.79, 20.69, 20.67, 20.89)
pehe_std <- c(2.32, 1.71, 1.87, 2.00, 1.80, 1.79, 1.86, 1.95, 1.97, 2.00, 1.84)

pehe_low <- pehe - pehe_std
pehe_high <- pehe + pehe_std

ate <- c(9.38, 8.14, 7.62, 7.54, 6.98, 6.72, 6.66, 6.72, 6.39, 6.50, 6.51)
ate_std <- c(1.92, 1.42, 1.52, 1.84, 1.40, 1.70, 1.65, 1.53, 1.68, 1.83, 1.52)

ate_low <- ate - ate_std
ate_high <- ate + ate_std

color_1 <- '#F59799'
color_2 <- '#9DC7EA'

par(mar = c(5.1, 5.6, 4.1, 5.6))
plot(percentages, ate_high, type = 'n', ylim = c(min(ate_low), max(ate_high)), cex.axis=2, ann=FALSE, mgp=c(3, 1, 0),
     xaxs = "i", yaxs = "i")
mtext(side = 1, text = "Percentage of Matches in Batch [%]", line = 3.75, cex=3)
mtext(side = 2, text = TeX("$\\epsilon_{ATE}$"), line = 2.75, cex=3, col=color_1)

polygon(c(percentages, rev(percentages)), c(ate_high, rev(ate_low)),
        col=makeTransparent(color_1), border = NA)

lines(percentages, ate, pch=4, type="o", cex=3, col=color_1, lwd=5)
lines(percentages, ate_low, lwd=1, lty="dashed", col=color_1)
lines(percentages, ate_high, lwd=1, lty="dashed", col=color_1)

par(new = T)
plot(percentages, axes=F, pehe_high, , type = 'n', ylim = c(min(pehe_low), max(pehe_high)), cex.axis=2, ann=FALSE,
     mgp=c(0, 0, 0), xaxs = "i", yaxs = "i")
axis(side=4, mgp=c(3, 1.5, 0), cex.axis=2, yaxs = "i")
mtext(side=4, text=TeX("$\\sqrt{\\epsilon_{PEHE}}$"), cex=3, line=4, col=color_2)


polygon(c(percentages, rev(percentages)), c(pehe_high, rev(pehe_low)),
        col=makeTransparent(color_2), border = NA)
lines(percentages, pehe, pch=1, type="o", cex=3, col=color_2, lwd=5)
lines(percentages, pehe_low, lwd=1, lty="dashed", col=color_2)
lines(percentages, pehe_high, lwd=1, lty="dashed", col=color_2)
dev.off()