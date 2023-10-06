install.packages("hrbrthemes")
devtools::install_github("mattflor/chorddiag")
install.packages("chorddiag")


# Libraries
library(tidyverse)
library(viridis)
library(patchwork)
library(hrbrthemes)
library(circlize)
library(chorddiag)  #devtools::install_github("mattflor/chorddiag")

# Load dataset from github
#data <- read.table("/Users/dglenn/Desktop/ChordDiagram/MinJanTempRound5-0509.csv", header=TRUE)
#data <- read.table("/Users/dglenn/Desktop/ChordDiagram/MinJanTempRound5.csv", header=TRUE)
#data <- read.table("/Users/dglenn/Desktop/ChordDiagram/WhereColdGoes.csv", header=TRUE)
data <- read.table("/Users/dglenn/Desktop/ChordDiagram/MinJanTempBins.csv", header=TRUE)
#data <- read.table("/Users/dglenn/Desktop/ChordDiagram/MinJanTempBins-0509.csv", header=TRUE)
#numcat<-17 #number of categories
#numcat<-15 #number of categories
numcat<-5 #number of categories

# short names

#colnames(data) <- c('-20','-15','-10','-5',	'0', '5', '10', '15', '20', '25', '30',	'35',	'40',	'45',	'50',	'55',	'60')
#colnames(data) <- c('-10',	'-5',	'0', '5', '10', '15', '20', '25', '30',	'35',	'40',	'45',	'50',	'55',	'60')
colnames(data) <- c("<10","10s","20s","30s","40+")
rownames(data) <- colnames(data)

# I need a long format
data_long <- data %>%
  rownames_to_column %>%
  gather(key = 'key', value = 'value', -rowname)

# parameters
circos.clear()
circos.par(start.degree = 90, gap.degree = 4, track.margin = c(-0.1, 0.1), points.overflow.warning = FALSE)
par(mar = rep(0, 4))

# color palette
mycolor <- viridis(numcat, alpha = 1, begin = 0, end = 1, option = "D")
mycolor <- mycolor[sample(1:numcat)]
mycolor <- c("#481567FF","#2D708EFF","#29AF7FFF","#95D840FF","#FDE725FF")
mycolor <- c("#0066cc","#339966","#FDE725FF","#ffa64d","#ff0000")

# Base plot
chordDiagram(
  x = data_long, 
  grid.col = mycolor,
  transparency = 0.25,
  directional = 1,
  direction.type = c("arrows", "diffHeight"), 
  diffHeight  = -0.04,
  annotationTrack = "grid", 
  annotationTrackHeight = c(0.05, 0.1),
  link.arr.type = "big.arrow", 
  link.sort = TRUE, 
  link.largest.ontop = TRUE)

# Add text and axis
circos.trackPlotRegion(
  track.index = 1, 
  bg.border = NA, 
  panel.fun = function(x, y) {
    
    xlim = get.cell.meta.data("xlim")
    sector.index = get.cell.meta.data("sector.index")
    
    # Add names to the sector. 
    circos.text(
      x = mean(xlim), 
      y = 3.2, 
      labels = sector.index, 
      facing = "bending", 
      cex = 0.8
    )
    
    # Add graduation on axis
    #circos.axis(
      #h = "top", 
      #major.at = seq(from = 0, to = xlim[2], by = ifelse(test = xlim[2]>10, yes = 2, no = 1)), 
      #minor.ticks = 1, 
      #major.tick.percentage = 0.5,
      #labels.niceFacing = FALSE)
  }
)

