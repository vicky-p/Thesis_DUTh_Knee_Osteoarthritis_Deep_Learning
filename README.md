# Thesis_DUTh_Knee_Osteoarthritis_Deep_Learning
Κώδικας Διπλωματικής Εργασίας «Ανάλυση ακτινογραφιών γονάτου με εφαρμογές στην πρόγνωση και επιτήρηση της οστεοαρθρίτιδας». Δημοκρίτειο Πανεπιστήμιο Θράκης - Τμήμα Ηλεκτρολόγων Μηχανικών και Μηχανικών Υπολογιστών

# Περίληψη (Ελληνικά)
Σκοπός της παρούσας Διπλωματικής Εργασίας είναι η ανάπτυξη, υλοποίηση και η αξιολόγηση μεθόδων αυτόματης βαθμολόγησης ακτινογραφιών γόνατος για τη διάγνωση και παρακολούθηση της οστεοαρθρίτιδας (ΟΑ) του γόνατος. Πρωταρχικός στόχος είναι η αντιμετώπιση της διαγνωστικής υποκειμενικότητας που χαρακτηρίζει τη χειροκίνητη ιατρική εκτίμηση, αντικαθιστώντας τη  με έναν αντικειμενικό αυτόματο σύστημα ταξινόμησης, που βαθμολογεί τα ακτινολογικά ευρήματα σύμφωνα με την κλίμακα Kellgren-Lawrence (KL βαθμίδες 0–4) 

Ως σημείο σύγκρισης χρησιμοποιείται η εργασία της Thomson et al. (2016), ένα ιδιαίτερα ολοκληρωμένο δημοσιευμένο σύστημα με χειροκίνητα χαρακτηριστικά, το οποίο πέτυχε AUC=0.904 για binary OA detection και weighted kappa=0.49 για πολυκατηγορική βαθμολόγηση KL 0–4, βασισμένο σε παραδοσιακές τεχνικές μηχανικής όρασης. Η μέθοδος προσέγγισης της παρούσας εργασίας είναι κλιμακωτή και περιλαμβάνει: 

(α) Την ανάπτυξη ενός pipeline χειροκίνητα εξαγώμενων χαρακτηριστικών και ταξινόμηση Random Forest ως baseline σύστημα (Φάση 1α και Φάση 1β).

(β) Τη μετάβαση σε σύγχρονες αρχιτεκτονικές Βαθιάς Μάθησης με την υλοποίηση pipeline με Vision Transformer ViT-B/16 (Φάση 2α),

(γ) Τη διεξαγωγή εκτενών πειραμάτων με Convolutional Neural Networks (ResNet-50, EfficientNet-B4), Υβριδικά μοντέλα, τεχνικές Test-Time Augmentation (TTA) και συνδυαστικές μεθόδους Ensemble (Φάση 2β). 

(δ) Όλα τα μοντέλα εκπαιδεύτηκαν και αξιολογήθηκαν στο δημόσιο σύνολο δεδομένων "Knee OA Dataset with Severity" (7260 ακτινογραφίες, υποσύνολο του OAI), και

(ε) Την παροχή ερμηνευσιμότητας των αποφάσεων των συστημάτων μέσω του αλγορίθμου Grad-CAM.[

Τα κύρια αποτελέσματα αποδεικνύουν την απόλυτη υπεροχή της Βαθιάς Μάθησης. Ο ViT-B/16 πέτυχε kw=0.816 (+67% έναντι του Thomson baseline) και AUC=0.925 (+2.3%). Το βέλτιστο αποτέλεσμα επιτεύχθηκε μέσω της συνδυαστικής μεθόδου Ensemble (ViT + ResNet), αγγίζοντας kw=0.826 και AUC=0.932.χωρίς επιπλέον εκπαίδευση. Η ισχυρή στατιστική ανάλυση (paired bootstrap, 2.000 resamples) αποδεικνύει αφενός την ισοδυναμία μεταξύ ViT-B/16 και ResNet-50 είναι στατιστικά ισοδύναμα (P=0.527), και αφετέρου επιβεβαιώνει τη σημαντική υπεροχή του μοντέλου Ensemble έναντι όλων (P=0.999).

Επιπλέον, η έρευνα φέρνει στο φως δύο κρίσιμα ευρήματα: 
1. Πρώτον, τα Grad-CAM heatmaps επιβεβαιώνουν ότι τα μοντέλα λαμβάνουν ανατομικά ορθές αποφάσεις, εστιάζοντας σωστά στα κρίσιμα σημεία της εικόνας. Αποκαλύπτουν, δε, συμπληρωματικά attention patterns: το ViT εστιάζει κυκλικά στα οστεόφυτα (spot-like), ενώ το ResNet-50 γραμμικά στον αρθρικό χώρο (band-like), αιτιολογώντας έτσι την υπεροχή του μοντέλου Ensemble.
2. Δεύτερον, εντοπίστηκε ένα ισχυρό φαινόμενο "θορύβου ετικετών" (label noise) στην ενδιάμεση κλάση KL2 grade, όπου το 28.2% των ακτινογραφιών στάθηκε αδύνατο να ταξινομηθεί σωστά από όλα τα μοντέλα ανεξαιρέτως. Το εύρημα αυτό αποδεικνύει ότι η περαιτέρω βελτίωση της διάγνωσης στα αρχικά στάδια της νόσου δεν περιορίζεται από την αλγοριθμική ισχύ, αλλά από την εγγενή ιατρική υποκειμενικότητα των αρχικών δεδομένων (πρόβλημα annotations).

# Λέξεις-Κλειδιά: 
Οστεοαρθρίτιδα Γόνατος, Κλίμακα Kellgren-Lawrence, Vision Transformer, ResNet-50, Μεταφορά Μάθησης (Transfer Learning), Grad-CAM, Ensemble, Βαθιά Μάθηση, Ιατρική Επεξεργασία Εικόνας.

# Abstract (English)
The purpose of this diploma thesis is the development, implementation, and evaluation of automated methods for grading knee radiographs in the diagnosis and monitoring of knee Osteoarthritis (OA). The primary objective is to address the diagnostic subjectivity inherent in manual clinical assessments by replacing them with an objective, automated classification system that grades radiological findings according to the Kellgren-Lawrence scale (KL grades 0–4).
The work of Thomson [2] serves as the benchmark, representing a highly comprehensive published system using hand-crafted features based on traditional computer vision techniques, which achieved AUC=0.904 for binary OA detection and weighted kappa=0.49 for multiclass KL 0–4 grading. The methodological approach of this thesis is progressive and comprises of:

(a) The development of a hand-crafted feature extraction pipeline with a Random Forest classifier as a baseline system (Phase 1a and Phase 1b).

(b) The transition to modern Deep Learning architectures through the implementation of a pipeline featuring the Vision Transformer ViT-B/16 (Phase 2a).

(c) Extensive comparative experiments employing Convolutional Neural Networks (ResNet-50, EfficientNet-B4), Hybrid models, Test-Time Augmentation (TTA) techniques, and Ensemble methods (Phase 2b).

(d) Model training and evaluation on the public "Knee OA Dataset with Severity" (7260 radiographs, a subset of the OAI dataset).

(e) Decision interpretability for the systems using the Grad-CAM algorithm.

The main results demonstrate the clear superiority of Deep Learning. ViT-B/16 achieved 𝑘𝑤=0.816 (+67% over the Thomson baseline) and AUC=0.925 (+2.3%). The optimal performance was attained via the ViT + ResNet Ensemble combination, reaching 𝑘𝑤=0.826 and AUC=0.932 without requiring additional training. Robust statistical analysis (paired bootstrap, 2,000 resamples) demonstrates that ViT-B/16 and ResNet-50 are statistically equivalent (𝑃=0.527), while confirming the significant superiority of the Ensemble model over all individual models (𝑃=0.999).

Furthermore, this research brings to light two critical findings:
1. Grad-CAM heatmaps confirm that the models make anatomically correct decisions, focusing appropriately on critical image regions. Moreover, they reveal complementary attention patterns: ViT displays a spot-like focus on osteophytes, whereas ResNet-50 shows a band-like attention along the joint space, thus providing mathematical and physical justification for the Ensemble model's superiority.
2. A pronounced label noise phenomenon was identified in the intermediate KL2 grade, where 28.2% of the radiographs could not be correctly classified by any model. This finding proves that further diagnostic improvement in early-stage disease is bound not by algorithmic capacity, but by the inherent medical subjectivity and annotation noise present in the ground-truth data.

# Keywords: 
Knee Osteoarthritis, Kellgren-Lawrence Scale, Vision Transformer, ResNet-50, Transfer Learning, Grad-CAM, Ensemble Methods, Deep Learning, Medical Image Analysis.
