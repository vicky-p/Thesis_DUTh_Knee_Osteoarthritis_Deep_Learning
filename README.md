# Thesis_DUTh_Knee_Osteoarthritis_Deep_Learning
Κώδικας Διπλωματικής Εργασίας «Ανάλυση ακτινογραφιών γονάτου με εφαρμογές στην πρόγνωση και επιτήρηση της οστεοαρθρίτιδας». Δημοκρίτειο Πανεπιστήμιο Θράκης - Τμήμα Ηλεκτρολόγων Μηχανικών και Μηχανικών Υπολογιστών

# Περίληψη (Ελληνικά)
Η παρούσα Διπλωματική Εργασία εστιάζει στην ανάπτυξη, υλοποίηση και συστηματική αξιολόγηση καινοτόμων μεθόδων τεχνητής νοημοσύνης για την αυτόματη σταδιοποίηση της Οστεοαρθρίτιδας (ΟΑ) γόνατος. Πρωταρχικός στόχος είναι η αντιμετώπιση της διαγνωστικής υποκειμενικότητας που χαρακτηρίζει τη χειροκίνητη ιατρική εκτίμηση, μέσω ενός αντικειμενικού συστήματος βαθμολόγησης των ακτινολογικών ευρημάτων βάσει της κλίμακας Kellgren-Lawrence (KL βαθμίδες 0–4).

Για την ακριβή μέτρηση της εξέλιξης των μεθόδων, το σύστημά μας συγκρίνεται με την καθιερωμένη εργασία της Thomson et al. (2016) (AUC=0.904, k_w=0.49), η οποία βασίστηκε σε παραδοσιακές τεχνικές μηχανικής όρασης. Η μεθοδολογική μας προσέγγιση είναι κλιμακωτή και περιλαμβάνει:
(α) Την ανάπτυξη ενός baseline συστήματος με χειροκίνητα εξαγόμενα χαρακτηριστικά και ταξινομητή Random Forest (Φάση 1α και Φάση 1β).
(β) Τη μετάβαση σε αρχιτεκτονικές Βαθιάς Μάθησης, με αιχμή τον σύγχρονο Vision Transformer ViT-B/16 (Φάση 2α).
(γ) Τη διεξαγωγή εκτενών πειραμάτων με Convolutional Neural Networks (ResNet-50, EfficientNet-B4), Υβριδικά μοντέλα, τεχνικές Test-Time Augmentation (TTA) και συνδυαστικές μεθόδους Ensemble (Φάση 2β).
Όλα τα μοντέλα εκπαιδεύτηκαν και αξιολογήθηκαν στο ανοιχτό σύνολο δεδομένων "Knee OA Dataset with Severity" (7.260 ακτινογραφίες, υποσύνολο του OAI), ενώ η ιατρική ερμηνευσιμότητα των αποφάσεων εξασφαλίστηκε μέσω του αλγορίθμου Grad-CAM.

Τα πειραματικά αποτελέσματα αποδεικνύουν την απόλυτη υπεροχή της Βαθιάς Μάθησης. Ο ViT-B/16 πέτυχε AUC=0.925 και σταθμισμένο Kappa k_w=0.816, σημειώνοντας εντυπωσιακή βελτίωση 67% έναντι του baseline. Το βέλτιστο αποτέλεσμα επιτεύχθηκε μέσω της συνδυαστικής μεθόδου Ensemble (ViT + ResNet), αγγίζοντας k_w=0.826 και AUC=0.932. Η ισχυρή στατιστική ανάλυση (paired bootstrap, 2.000 αναδειγματοληψίες) επιβεβαιώνει αφενός την ισοδυναμία μεταξύ ViT και ResNet-50 (P=0.527), και αφετέρου τη σημαντική υπεροχή του Ensemble έναντι όλων (P=0.999).

Επιπλέον, η έρευνα φέρνει στο φως δύο κρίσιμα ευρήματα: Πρώτον, οι χάρτες Grad-CAM αποδεικνύουν ότι τα μοντέλα λαμβάνουν ανατομικά ορθές αποφάσεις, αξιοποιώντας συμπληρωματικά οπτικά μοτίβα (το ViT εστιάζει κυκλικά στα οστεόφυτα, ενώ το ResNet-50 γραμμικά στον αρθρικό χώρο). Δεύτερον, εντοπίστηκε ένα ισχυρό φαινόμενο "θορύβου ετικετών" (label noise) στην ενδιάμεση κλάση KL2, όπου το 28.2% των ακτινογραφιών στάθηκε αδύνατο να ταξινομηθεί σωστά από όλα τα μοντέλα ανεξαιρέτως. Το εύρημα αυτό αποδεικνύει ότι η περαιτέρω βελτίωση της διάγνωσης στα αρχικά στάδια της νόσου δεν περιορίζεται από την αλγοριθμική ισχύ, αλλά από την εγγενή ιατρική υποκειμενικότητα των αρχικών δεδομένων.

# Λέξεις-Κλειδιά: 
Οστεοαρθρίτιδα Γόνατος, Κλίμακα Kellgren-Lawrence, Vision Transformer, ResNet-50, Μεταφορά Μάθησης (Transfer Learning), Grad-CAM, Ensemble, Βαθιά Μάθηση, Ιατρική Επεξεργασία Εικόνας.

# Abstract (English)
This diploma thesis focuses on the development, implementation, and rigorous evaluation of advanced artificial intelligence methodologies for the automated staging of knee Osteoarthritis (OA). The primary objective is to overcome the diagnostic subjectivity inherent in manual clinical assessments by introducing an objective, deep learning-driven classifier that grades radiological findings according to the Kellgren-Lawrence (KL) scale (grades 0–4).

To accurately measure algorithmic progress, our pipeline is benchmarked against the established work of Thomson et al. (2016), a comprehensive system based on hand-crafted features (AUC=0.904, k_w=0.49). Our methodological approach is progressive and includes:
(a) A baseline pipeline utilizing hand-crafted feature extraction and a Random Forest classifier (Phase 1a & Phase 1b).
(b) A transition to modern Deep Learning architectures, spearheaded by the Vision Transformer ViT-B/16 (Phase 2a).
(c) Extensive comparative experiments employing Convolutional Neural Networks (ResNet-50, EfficientNet-B4), Hybrid architectures, Test-Time Augmentation (TTA), and Ensemble methods (Phase 2b).
All models were trained and evaluated on the public "Knee OA Dataset with Severity" (7,260 radiographs, an OAI subset). Furthermore, the medical interpretability of the models' decisions was validated using the Grad-CAM algorithm. 

The experimental results demonstrate the absolute superiority of Deep Learning. The ViT-B/16 achieved an AUC=0.925 and a weighted Kappa of k_w=0.816, marking a remarkable 67% improvement over the classical baseline. The optimal overall performance was achieved by the ViT+ResNet Ensemble, reaching k_w=0.826 and AUC=0.932 without additional training. Robust statistical analysis (paired bootstrap, 2,000 resamples) confirms that while ViT-B/16 and ResNet-50 are statistically equivalent (P=0.527), the Ensemble significantly outperforms both (P=0.999).

Crucially, this research highlights two major findings: First, Grad-CAM heatmaps verify that the models focus on anatomically correct regions, revealing complementary attention patterns—ViT exhibits a spot-like focus on osteophytes, whereas ResNet displays band-like attention along the joint space—which mathematically justifies the Ensemble's superiority. Second, a severe label noise phenomenon was identified in the borderline KL2 grade, where 28.2% of the images were consistently misclassified by all evaluated models. This finding proves that further improvements in early-stage OA diagnosis are currently bounded not by algorithmic capacity, but by the inherent medical subjectivity and annotation noise present in the ground-truth data.

# Keywords: 
Knee Osteoarthritis, Kellgren-Lawrence Scale, Vision Transformer, ResNet-50, Transfer Learning, Grad-CAM, Ensemble Methods, Deep Learning, Medical Image Analysis.
