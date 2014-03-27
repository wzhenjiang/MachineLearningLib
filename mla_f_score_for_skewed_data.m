function f_score = mla_f_score_for_skewed_data(true_postive, num_predicted_positive, num_actual_postive)
%% Purpose: this function is used to compute f_score for skewed data

P = true_positive / num_predicted_positive;
R = true_positive / num_actual_positive;
f_score = 2 * P * R / (P + R);

end