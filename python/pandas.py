import pandas as pd

def scored_merge(left:pd.DataFrame, right:pd.DataFrame, merge_keys:dict, suffix: list = ['left','rigth'], min_score: int = 1):
    """
    Performs a tiered merge between two DataFrames with match scoring based on key intersections.
    
    The function executes a series of inner joins starting with all specified keys,
    then progressively reducing the number of keys used, and finally combines all results
    with a score column indicating how many keys matched for each row.

    Parameters
    ----------
    left : pd.DataFrame
        Left DataFrame to merge
    right : pd.DataFrame
        Right DataFrame to merge
    merge_keys : dict
        Dictionary mapping key columns between DataFrames in format {left_key: right_key}
    suffix : list, optional
        Suffixes to append to overlapping columns (default: ['_left', '_right'])
    min_score : int, optional
        Minimum number of key matches required to include in results (default: 1)

    Returns
    -------
    pd.DataFrame
        A combined DataFrame containing all successful merges with additional columns:
        - merge_score: Number of keys that matched (ranges from min_score to total keys)
        - Original columns with suffixes applied to overlapping non-key columns

    Examples
    --------
    >>> df1 = pd.DataFrame({'id': [1, 2], 'code': ['A', 'B'], 'value': [100, 200]})
    >>> df2 = pd.DataFrame({'id': [1, 2], 'code': ['A', 'C'], 'category': ['X', 'Y']})
    >>> result = scored_merge(
    ...     df1, df2,
    ...     merge_keys={'id': 'id', 'code': 'code'},
    ...     min_score=1
    ... )
    >>> print(result)
       id code_left value code_right category  merge_score
    0   1         A   100          A        X            2
    1   2         B   200          C        Y            1

    Notes
    -----
    1. The merge is performed as a series of inner joins from most to least specific
    2. Rows with higher merge_score represent more exact matches
    3. For large DataFrames, consider filtering by min_score first for better performance
    """
    merge_result = []
    n_keys = len(merge_keys)
    left_keys = list(merge_keys.keys())
    right_keys = list(merge_keys.values())
    
    for i in range(n_keys,min_score, -1):
        partial_merge = pd.merge(left, right, left_on= left_keys[:i], right_on= right_keys[:i], how="inner", suffixes=suffix)
        partial_merge['merge_score'] = i
        merge_result.append(partial_merge)
    
    final_result = pd.concat(merge_result, ignore_index=True)
    return final_result
