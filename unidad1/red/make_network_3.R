library(DESeq2)       # if you need it elsewhere
library(WGCNA)
library(pheatmap)
library(dynamicTreeCut)
library(grid)

options(stringsAsFactors = FALSE)
allowWGCNAThreads()

dir.create("../output", showWarnings = FALSE, recursive = TRUE)

## ---- 1) Read expression (robust to wrong sep) ----
# Try comma CSV (most likely), fallback to semicolon
expr_path <- "expression_from_paperIDs_for_WGCNA.csv"

counts <- tryCatch(
  read.csv(expr_path, header = TRUE, row.names = 1, check.names = FALSE),
  error = function(e) NULL
)
if (is.null(counts) || ncol(counts) == 1) {
  counts <- read.csv2(expr_path, header = TRUE, row.names = 1, check.names = FALSE)
}

# Coerce to numeric safely
counts[] <- lapply(counts, function(x) suppressWarnings(as.numeric(as.character(x))))
# Drop rows (genes) that became all NA
counts <- counts[rowSums(is.finite(as.matrix(counts))) > 0, , drop = FALSE]

## ---- 2) Read metadata and align ----
metadata <- read.csv("metadata.csv", header = TRUE, row.names = 1, check.names = FALSE)
# Keep samples present in both
commonSamples <- intersect(colnames(counts), rownames(metadata))
if (length(commonSamples) < 2) stop("Not enough overlapping samples between counts and metadata.")
counts   <- counts[, commonSamples, drop = FALSE]
metadata <- metadata[commonSamples, , drop = FALSE]

## ---- 3) Build WGCNA expression (samples x genes) & QC ----
datExpr0 <- t(as.matrix(counts))          # samples x genes
storage.mode(datExpr0) <- "double"

# Basic cleanup: set non-finite to NA
datExpr0[!is.finite(datExpr0)] <- NA

# Remove obviously bad samples/genes before any clustering
gsg <- goodSamplesGenes(datExpr0, verbose = 3)
if (!gsg$allOK) {
  datExpr0 <- datExpr0[gsg$goodSamples, gsg$goodGenes, drop = FALSE]
  metadata <- metadata[rownames(datExpr0), , drop = FALSE]
}

nGenes   <- ncol(datExpr0)
nSamples <- nrow(datExpr0)

print("hola")
colnames(metadata)

## ---- 4) Sample clustering (use correlation distance, robust to some NAs) ----
dissSamples <- as.dist(1 - cor(t(datExpr0), use = "pairwise.complete.obs"))
sampleTree  <- hclust(dissSamples, method = "average")

svg("../output/sample_clustering.svg", width = 8, height = 6)
par(cex = 0.7, mar = c(0,4,2,0))
plot(sampleTree, main = "Sample clustering to detect outliers", sub = "", xlab = "")
abline(h = 0, col = "red")  # change if you want to cut
dev.off()

# Optionally cut: here we keep all (cutHeight = 0 means no cut)
clust <- cutreeStatic(sampleTree, cutHeight = 0, minSize = 1)
#keepSamples <- (clust == 1)
#datExpr0 <- datExpr0[keepSamples, , drop = FALSE]

keepSamples <- rep(TRUE, nrow(datExpr0))
datExpr0 <- datExpr0[keepSamples, , drop = FALSE]
metadata <- metadata[keepSamples, , drop = FALSE]
nSamples <- nrow(datExpr0)

## ---- 5) Traits (make numeric coding for tissue) ----
datTraits <- metadata[, c("tissue:ch1","age:ch1"), drop = FALSE]
datTraits[,"tissue:ch1"] <- ifelse(datTraits$"tissue:ch1" %in% c("mucosa","normal","adjacent normal"), 0, 1)
rownames(datTraits) <- rownames(datExpr0)

sampleTree2  <- hclust(as.dist(1 - cor(t(datExpr0), use = "pairwise.complete.obs")), method = "average")
traitColors  <- numbers2colors(datTraits, signed = FALSE)

svg("../output/dendrogram_plot.svg", width = 8, height = 6)
plotDendroAndColors(sampleTree2, traitColors,
                    groupLabels = colnames(datTraits),
                    main = "Sample dendrogram and trait heatmap")
dev.off()

## ---- 6) Soft-threshold selection ----
powers <- 1:20
sft <- pickSoftThreshold(datExpr0, powerVector = powers, networkType = "signed",
                         corFnc = "bicor", corOptions = list(use = "pairwise.complete.obs"),
                         verbose = 5)

svg("../output/scale_independence_and_mean_connectivity.svg", width = 10, height = 5)
par(mfrow = c(1,2))
plot(sft$fitIndices[,1],
     -sign(sft$fitIndices[,3]) * sft$fitIndices[,2],
     xlab = "Soft Threshold (power)", ylab = "Scale Free Topology Model Fit",
     main = "Scale independence", type = "n")
text(sft$fitIndices[,1],
     -sign(sft$fitIndices[,3]) * sft$fitIndices[,2],
     labels = sft$fitIndices[,1])
abline(h = 0.8, col = "red", lty = 2)
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab = "Soft Threshold (power)", ylab = "Mean connectivity",
     main = "Mean connectivity", type = "n")
text(sft$fitIndices[,1], sft$fitIndices[,5], labels = sft$fitIndices[,1])
dev.off()

softPower <- ifelse(is.na(sft$powerEstimate), 14, sft$powerEstimate)
message("Using softPower = ", softPower)

## ---- 7) Network construction (custom path, like yours) ----
adjacency <- adjacency(datExpr0, power = softPower, type = "signed",
                       corFnc = "bicor", corOptions = list(use = "pairwise.complete.obs"))
degree <- rowSums(adjacency)

svg("../output/degree_distribution.svg", width = 8, height = 6)
hist(degree, breaks = 50, col = "gray",
     main = "Degree Distribution", xlab = "Degree", ylab = "Number of Genes")
dev.off()

TOM    <- TOMsimilarity(adjacency, TOMType = "signed")
dissTOM <- 1 - TOM
diag(dissTOM) <- 0

geneTree <- hclust(as.dist(dissTOM), method = "average")

dynamicMods <- cutreeDynamic(dendro = geneTree, distM = dissTOM,
                             deepSplit = 2, pamRespectsDendro = FALSE,
                             minClusterSize = 30)
dynamicColors <- labels2colors(dynamicMods)

MEList <- moduleEigengenes(datExpr0, colors = dynamicColors)
MEs    <- orderMEs(MEList$eigengenes)

MEDiss <- 1 - cor(MEs, use = "pairwise.complete.obs")
METree <- hclust(as.dist(MEDiss), method = "average")

svg("../output/clustering_of_module_eigengenes.svg", width = 8, height = 6)
plot(METree, main = "Clustering of Module Eigengenes", xlab = "", sub = "")
abline(h = 0, col = "red")
dev.off()

merged <- mergeCloseModules(datExpr0, dynamicColors, cutHeight = 0, verbose = 3)
mergedColors <- merged$colors
mergedMEs    <- orderMEs(merged$newMEs)
moduleColors <- mergedColors
colorOrder   <- c("grey", standardColors(50))
moduleLabels <- match(moduleColors, colorOrder) - 1
MEs          <- mergedMEs

svg("../output/TOMplot.svg", width = 8, height = 8)
TOMplot(dissTOM, geneTree, moduleColors, main = "TOMplot")
dev.off()

## ---- 8) Module-trait relationships ----
MEs0 <- moduleEigengenes(datExpr0, moduleColors)$eigengenes
MEs  <- orderMEs(MEs0)

moduleTraitCor    <- cor(MEs, datTraits, use = "pairwise.complete.obs")
moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nSamples)

textMatrix <- paste(signif(moduleTraitCor, 2), "\n(",
                    signif(moduleTraitPvalue, 1), ")", sep = "")
dim(textMatrix) <- dim(moduleTraitCor)

svg("../output/moduletrait_relationships.svg", width = 9, height = 6)
labeledHeatmap(Matrix = moduleTraitCor,
               xLabels = colnames(datTraits),
               yLabels = colnames(MEs),
               colorLabels = FALSE,
               colors = blueWhiteRed(50),
               textMatrix = textMatrix,
               setStdMargins = FALSE,
               cex.text = 0.7,
               zlim = c(-1, 1),
               main = "Module-trait relationships")
dev.off()

## ---- 9) MM vs GS plots ----
nSamples <- nrow(datExpr0)
tumoral  <- data.frame(Tumoral = datTraits$"tissue:ch1")
modNames <- substring(colnames(MEs), 3)

geneModuleMembership <- as.data.frame(cor(datExpr0, MEs, use = "pairwise.complete.obs"))
MMPvalue             <- as.data.frame(corPvalueStudent(as.matrix(geneModuleMembership), nSamples))
names(geneModuleMembership) <- paste0("MM", modNames)

geneTraitSignificance <- as.data.frame(cor(datExpr0, tumoral, use = "pairwise.complete.obs"))
GSPvalue              <- as.data.frame(corPvalueStudent(as.matrix(geneTraitSignificance), nSamples))
names(geneTraitSignificance) <- paste0("GS.", names(tumoral))

dir.create("../output/mm_vs_gs_plots", showWarnings = FALSE)
for (module in setdiff(unique(moduleColors), "grey")) {
  column <- match(module, modNames)
  if (is.na(column)) next
  moduleGenes <- moduleColors == module
  svg(paste0("../output/mm_vs_gs_plots/", module, "_membership_vs_gs.svg"), width = 8, height = 6)
  verboseScatterplot(abs(geneModuleMembership[moduleGenes, column]),
                     abs(geneTraitSignificance[moduleGenes, 1]),
                     xlab = paste("Module Membership in", module),
                     ylab = "Gene Significance (Tumoral)",
                     main = paste("MM vs GS -", module),
                     col = module)
  dev.off()
}

## ---- 10) MDS of genes (optional; can be heavy) ----
datExpr.genes <- t(datExpr0)
gene.dist <- dist(datExpr.genes)
mds.out <- cmdscale(gene.dist, k = 2)
gene.colors <- labels2colors(moduleColors)

svg("../output/mds_plot.svg", width = 8, height = 6)
plot(mds.out, col = gene.colors, pch = 19,
     main = "MDS plot (genes)", xlab = "MDS1", ylab = "MDS2")
dev.off()

## ---- 11) Save module assignment & counts ----
write.csv(data.frame(Gene = colnames(datExpr0), Module = moduleColors),
          "../output/gene_module_membership.csv", row.names = FALSE)
module_gene_counts <- as.data.frame(table(moduleColors))
colnames(module_gene_counts) <- c("Module", "GeneCount")
write.csv(module_gene_counts, "../output/module_gene_counts.csv", row.names = FALSE)

## ---- 12) Export per-module networks for Cytoscape (like before) ----
dir.create("../output/cytoscape_all_modules", showWarnings = FALSE)
for (module in setdiff(unique(moduleColors), "grey")) {
  inModule <- (moduleColors == module)
  modGenes <- colnames(datExpr0)[inModule]
  if (length(modGenes) < 2) next
  modExpr <- datExpr0[, modGenes, drop = FALSE]
  adjM <- adjacency(modExpr, power = softPower, type = "signed",
                    corFnc = "bicor", corOptions = list(use = "pairwise.complete.obs"))
  TOMm <- TOMsimilarity(adjM, TOMType = "signed")
  rownames(TOMm) <- colnames(TOMm) <- modGenes
  exportNetworkToCytoscape(
    TOMm,
    edgeFile = paste0("../output/cytoscape_all_modules/edges_", module, ".txt"),
    nodeFile = paste0("../output/cytoscape_all_modules/nodes_", module, ".txt"),
    weighted = TRUE,
    threshold = 0.02,
    nodeNames = modGenes
  )
}
