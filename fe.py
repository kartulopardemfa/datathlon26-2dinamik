import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

SKILLS = ['coding_score','problem_solving_score','data_structures_score','sql_score',
          'machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']

ROLE_SKILLS = {
 'Backend Developer': ['backend_score','sql_score','data_structures_score','coding_score'],
 'Frontend Developer': ['frontend_score','coding_score','problem_solving_score'],
 'Software Developer': ['coding_score','data_structures_score','problem_solving_score','backend_score','frontend_score'],
 'Data Scientist': ['machine_learning_score','sql_score','problem_solving_score'],
 'Data Analyst': ['sql_score','machine_learning_score','problem_solving_score'],
 'AI Engineer': ['machine_learning_score','coding_score','data_structures_score'],
 'Cloud Engineer': ['cloud_score','devops_score','backend_score'],
 'DevOps Engineer': ['devops_score','cloud_score','backend_score'],
 'Machine Learning Engineer': ['machine_learning_score','coding_score','data_structures_score'],
 'Full Stack Developer': ['backend_score','frontend_score','coding_score','sql_score'],
 'Mobile Developer': ['coding_score','frontend_score','problem_solving_score'],
}

def engineer(df):
    df = df.copy()
    df['skill_mean'] = df[SKILLS].mean(axis=1)
    df['skill_max'] = df[SKILLS].max(axis=1)
    df['skill_min'] = df[SKILLS].min(axis=1)
    df['skill_std'] = df[SKILLS].std(axis=1)
    # role-matched skills
    rm = np.zeros(len(df)); rmax = np.zeros(len(df))
    for role, cols in ROLE_SKILLS.items():
        m = (df['target_role']==role).values
        if m.sum():
            rm[m] = df.loc[m, cols].mean(axis=1)
            rmax[m] = df.loc[m, cols].max(axis=1)
    df['role_skill_mean'] = rm
    df['role_skill_max'] = rmax
    df['role_skill_gap'] = df['role_skill_mean'] - df['skill_mean']
    # interview / soft
    df['interview_mean'] = df[['technical_interview_score','hr_interview_score']].mean(axis=1)
    df['soft_mean'] = df[['communication_score','teamwork_score','leadership_score','presentation_score']].mean(axis=1)
    # experience
    df['total_projects'] = df['real_client_project_count'] + df['freelance_project_count']
    df['exp_score'] = (df['real_client_project_count']*2 + df['freelance_project_count']
                       + df['internship_count'] + df['hackathon_awards']*2)
    df['internship_total'] = df['internship_count'] * df['internship_duration_months'].fillna(0)
    df['github_activity'] = df['github_repo_count'] * (1+df['github_avg_stars'].fillna(0))
    df['interview_ratio'] = df['interviews_attended'] / (df['applications_sent']+1)
    df['years_since_grad'] = df['application_year'] - df['graduation_year']
    # key interactions
    df['pq_x_ti'] = df['project_quality_score'] * df['technical_interview_score']
    df['pq_x_skill'] = df['project_quality_score'] * df['skill_mean']
    df['pq_x_role'] = df['project_quality_score'] * df['role_skill_mean']
    df['ti_x_skill'] = df['technical_interview_score'] * df['skill_mean']
    df['pq_x_comm'] = df['project_quality_score'] * df['communication_score']
    df['portfolio_x_github'] = df['portfolio_score'].fillna(0) * np.log1p(df['github_repo_count'])
    df['n_missing'] = df[['english_exam_score','internship_duration_months','portfolio_score',
                          'github_avg_stars','open_source_contribution_count',
                          'linkedin_profile_score','hr_interview_score']].isna().sum(axis=1)
    df['text_len'] = df['mentor_feedback_text'].str.len()
    df['text_words'] = df['mentor_feedback_text'].str.split().str.len()
    return df

def text_features(tr_txt, te_txt, y, n_svd=64, seed=42):
    """Returns (svd_tr, svd_te, oof_pred, te_pred) from TF-IDF."""
    tv = TfidfVectorizer(max_features=60000, ngram_range=(1,3), sublinear_tf=True, min_df=2)
    A = tv.fit_transform(pd.concat([tr_txt, te_txt]))
    Atr, Ate = A[:len(tr_txt)], A[len(tr_txt):]
    svd = TruncatedSVD(n_components=n_svd, random_state=seed)
    S = svd.fit_transform(A)
    Str, Ste = S[:len(tr_txt)], S[len(tr_txt):]
    # OOF ridge on tfidf
    kf = KFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(tr_txt)); te_pred = np.zeros(Ate.shape[0])
    for tr_i, va_i in kf.split(Atr):
        r = Ridge(alpha=1.0)
        r.fit(Atr[tr_i], y[tr_i])
        oof[va_i] = r.predict(Atr[va_i])
        te_pred += r.predict(Ate)/5
    return Str, Ste, oof, te_pred
