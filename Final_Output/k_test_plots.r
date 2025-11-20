library(tidyverse)
library(stringr)
library(patchwork)

# ==========================================
# PART 1: CIFAR-10 (Top Plot)
# ==========================================
setwd("/Users/mingikang/Developer/Convolutional-Nearest-Neighbor-Attention/Final_csv/")
df_full = read_csv("k_test.csv")

# ==========================================
# 1. HELPER FUNCTION
# ==========================================
create_vit_k_plot <- function(data, dataset_name, plot_title, show_x_label = TRUE) {
  
  # --- A. Filter Data for this Dataset ---
  df_subset <- data %>% filter(dataset == dataset_name)
  
  # --- B. Prepare Dynamic Lines (Traversing K=1-12) ---
  dynamic_lines <- df_subset %>%
    filter(
      (layer == "ConvNNAttention" & type == "all") |
        (layer == "ConvNNAttention" & type == "random") |
        (layer == "KvtAttention" & K <= 12)
    ) %>%
    filter(K %in% c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)) %>%
    mutate(Legend_Label = case_when(
      layer == "ConvNNAttention" & type == "all" ~ "ConvNN (All)",
      layer == "ConvNNAttention" & type == "random" ~ "ConvNN (Random)",
      layer == "KvtAttention" ~ "KvT"
    ))
  
  # --- C. Prepare Horizontal Lines (Constant Reference) ---
  hlines <- df_subset %>%
    filter(
      (layer == "Attention") |
        (layer == "KvtAttention" & K == 100) # <--- Fixed: CSV uses 'KvtAttention', not 'Kvt'
    ) %>%
    mutate(Legend_Label = case_when(
      layer == "Attention" ~ "Attention (Baseline)",
      layer == "KvtAttention" ~ "KvT (K=100)"
    ))
  
  # --- D. Create Plot ---
  p <- ggplot() +
    # 1. Dynamic Lines
    geom_line(
      data = dynamic_lines,
      aes(x = K, y = best_test_accuracy_top1, color = Legend_Label, linetype = "Dynamic"),
      linewidth = 1.0
    ) +
    
    # 2. Horizontal Lines
    geom_hline(
      data = hlines,
      aes(yintercept = best_test_accuracy_top1, color = Legend_Label, linetype = "Horizontal"),
      linewidth = 1.0
    ) +
    
    # 3. Scales
    scale_color_manual(
      name = "Model Type",
      values = c(
        "ConvNN (All)"         = "black",    # Black
        "ConvNN (Random)"      = "#377EB8",  # Blue
        "KvT"                  = "orange",  # Purple
        "KvT (K=100)"          = "orange",  # Purple (Baseline)
        "Attention (Baseline)" = "#4DAF4A"   # Green
      ),# IMPORTANT: Define the specific order so we can map linetypes correctly below
      breaks = c(
        "ConvNN (All)", 
        "ConvNN (Random)", 
        "KvT", 
        "KvT (K=100)", 
        "Attention (Baseline)"
      )
    ) +
    
    # --- NEW: Override the legend look ---
    guides(
      color = guide_legend(
        override.aes = list(
          # Map linetypes to the order defined in 'breaks' above:
          # 1. ConvNN (All)      -> solid
          # 2. ConvNN (Random)   -> solid
          # 3. KvT               -> solid
          # 4. KvT (K=100)       -> dotted
          # 5. Attention         -> dotted
          linetype = c("solid", "solid", "solid", "dotted", "dotted"),
          linewidth = c(1, 1, 1, 1, 1)
        )
      )
    ) +
    
    scale_linetype_manual(
      name = "Line Type",
      values = c("Dynamic" = "solid", "Horizontal" = "dotted"),
      guide = "none" 
    ) +
    
    scale_x_continuous(breaks = seq(2, 12, 1), limits = c(2, 12)) +
    
    # 4. Labels & Theme
    labs(
      title = plot_title,
      x = if(show_x_label) "K (Number of Neighbors)" else NULL,
      y = "Top-1 Accuracy (%)"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10)
    )
  
  return(p)
}

# ==========================================
# 2. GENERATE PLOTS
# ==========================================

# 1. CIFAR-10 (Top)
p_c10 <- create_vit_k_plot(df_full, "ViT-Tiny-CIFAR10", "CIFAR-10", show_x_label = FALSE)

# 2. CIFAR-100 (Bottom)
p_c100 <- create_vit_k_plot(df_full, "ViT-Tiny-CIFAR100", "CIFAR-100", show_x_label = TRUE)

# ==========================================
# 3. COMBINE & SAVE
# ==========================================
final_plot <- (p_c10 / p_c100) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(final_plot)

ggsave(
  "/Users/mingikang/Developer/Convolutional-Nearest-Neighbor-Attention/Final_Plots/ViT_k_plot.png",
  plot = final_plot,
  width = 8,
  height = 6,
  dpi = 500,
  bg = "white"
)