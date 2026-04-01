"""
TikTok API Configuration

Paste the tokens/cookies copied from browser Network tab here.
When tokens expire, copy new ones from the browser and update.
"""

# Values copied from browser (Network tab > Select request > Headers)
TIKTOK_CONFIG = {
    # Extracted from Request URL (Updated 2026-01-26)
    "msToken": "zoeENd_rOm8B4XsIoqAveXgFdtQzcoU1jjzTQdufkYTsaTgYSNBDuOHq6-MxCsOQprLF_FHhyVpBM8u9_L9Odb6o6tsxT8akF9hEdmEUeU7ZgUlHPJPFjlEYBvTKhROHh3ZUyEq1sGGf7TEb4XkbQ7Dn2Q==",
    "X-Bogus": "DFSzsIVO4ATANCc4Cu0OjxhGbwr1",
    "verifyFp": "verify_mkvfnki4_VNYk6yn6_ghvR_4x3l_AW4e_Ij2pUjDjA8la",
    "device_id": "7599721093834491423",
    "odinId": "7598233112162501645",
}

# Full cookie string (Updated 2026-03-07)
COOKIE_STRING = "_ttp=39zMqrsSFMYEZNZmgbStzM0Zy4s; tt_chain_token=urrHhPvTkKql3EVkbvUriw==; tiktok_webapp_theme_source=auto; tiktok_webapp_theme=dark; delay_guest_mode_vid=5; g_state={\"i_l\":0,\"i_ll\":1771873368909,\"i_b\":\"qTVqoKPC9rLPX6PzwDQH6rSYq4r/d/Hq9DqC+GoqC7Q\",\"i_e\":{\"enable_itp_optimization\":0}}; passport_csrf_token=3403e20834030ee4b59830d1a950f165; passport_csrf_token_default=3403e20834030ee4b59830d1a950f165; multi_sids=7598233112162501645%3A9f86edce6a0b3259b2196a907798adff; cmpl_token=AgQQAPNSF-RO0rlMtmWrPF008_09fzQX_4_ZYKEErA; passport_auth_status=f5968b1bfd20895700be0523e2023deb%2C; passport_auth_status_ss=f5968b1bfd20895700be0523e2023deb%2C; sid_guard=9f86edce6a0b3259b2196a907798adff%7C1771873446%7C15552000%7CSat%2C+22-Aug-2026+19%3A04%3A06+GMT; uid_tt=5f7032e60d131f9e9cfd4d7df85faf378286f0d5d8de773f7c9469d6cba3cad4; uid_tt_ss=5f7032e60d131f9e9cfd4d7df85faf378286f0d5d8de773f7c9469d6cba3cad4; sid_tt=9f86edce6a0b3259b2196a907798adff; sessionid=9f86edce6a0b3259b2196a907798adff; sessionid_ss=9f86edce6a0b3259b2196a907798adff; tt_session_tlb_tag=sttt%7C1%7Cn4btzmoLMlmyGWqQd5it___________D3FNJsebLchq_Pounz1Lar777Xr2VJo1b_xsAj0qYAWE%3D; sid_ucp_v1=1.0.1-KDUzNjljM2Q4ZWM5OWJiY2RmNzdlNzA0NTUxZmQ5NTQ4M2Q2NzQ4MDkKIQiNiLqYopiWuWkQpsnyzAYYswsgDDDGscnLBjgIQBJIBBAEGgd1c2Vhc3Q1IiA5Zjg2ZWRjZTZhMGIzMjU5YjIxOTZhOTA3Nzk4YWRmZjJNCiDCEFrMJIaCsibzFgDF9kcSMG-4h1qYlYbT_S805o5pNBIfhe6WkyRfQOyiVIDQ2UTb5HTXqPT5yw84DfxlATTrpBgEIgZ0aWt0b2s; ssid_ucp_v1=1.0.1-KDUzNjljM2Q4ZWM5OWJiY2RmNzdlNzA0NTUxZmQ5NTQ4M2Q2NzQ4MDkKIQiNiLqYopiWuWkQpsnyzAYYswsgDDDGscnLBjgIQBJIBBAEGgd1c2Vhc3Q1IiA5Zjg2ZWRjZTZhMGIzMjU5YjIxOTZhOTA3Nzk4YWRmZjJNCiDCEFrMJIaCsibzFgDF9kcSMG-4h1qYlYbT_S805o5pNBIfhe6WkyRfQOyiVIDQ2UTb5HTXqPT5yw84DfxlATTrpBgEIgZ0aWt0b2s; store-idc=useast5; store-country-code=us; store-country-code-src=uid; tt-target-idc=useast5; tt-target-idc-sign=DNn16t5yobRdIMlGcSUOOkoSul-4EX5lcZwUXFh7axZTEO0Gp_y5Ide9Y_SEXMoPBZ-y2_4wLLz0XOuqtjAcGO7zRGZQlvor9DUYIAbptdLKU8IuyS2PQ0-h-Uq2GIdMdSonO2oz3MaFP9iPxkt6MCtp0EftmT-0BzzYmeRNC4vs4H3DWRCaPhwyt_5ZihYjpYI2_nq4u4wPucYhi1QrB8ZjT_nsaWr833NS41rLkfUZENBVpJ3kC8PWZnIFCpAFj9MYVbxAZNL5uLZJH0khJbgDhbp_tEXmRezKrRfFoHuRe8QW9YN39B3_Tuwyu9VshO7umAzK0Q10-K13a-Jp4TT18_0IZk9v0sg-p13Bw6eHF_bOPHzJ9Oj1KmXTMAxSAEGY1KHYKqEI5ySOk8lGZYIJyQJUjgS-uIdauBfdlWJl55QuwdevcYFfCpZc8YigmTLbOQw1lvLTBdHZhIEBoRbzX7ksprvN-p_5ogzND2KHeYinTeImHFjk9N4_zpge; last_login_method=google; cookie-consent={%22optional%22:true%2C%22ga%22:true%2C%22af%22:true%2C%22fbp%22:true%2C%22lip%22:true%2C%22bing%22:true%2C%22ttads%22:true%2C%22reddit%22:true%2C%22hubspot%22:true%2C%22version%22:%22v10%22}; tt_csrf_token=Yz8TS50M-oVove9atm2bzjZwCBJTmnHatpd0; passport_fe_beating_status=true; ttwid=1%7CmiqP6kfnFNBYQRanoR_TI-F_3YRjf7ZwGcXqlJLgM5Q%7C1772878353%7C2aa108743e4e3ca1b5d5a0a0726471ef26b12d5fb2727bff7446a0493f7e7cc7; tt_ticket_guard_has_set_public_key=1; odin_tt=14a59157dfa62a3d9510f1f967415fa9d47ae6b47a2237a4a9cf84bc33512e35459f66ed5d00ded3e40bc2b9e3861401f1ad0159654c3b4ff165c241317d76d0d2e66146aeea20c8ff0b10fd8eac83ab; msToken=sIVwxe6_n2EjJi6xPZTjqHtvdkadqV5DhTmMnQAo3zreYudTdKmZu8SKksNXW8UixeMttDVd5DfAztB1tKHJzQlprLkFeL1YS3MtcgKcop0JZzIg1q74jVWyBerIldzhJONEM-SpxxY8K4xlA0gd3kg=; perf_feed_cache={%22expireTimestamp%22:1773050400000%2C%22itemIds%22:[%227612472436478446862%22%2C%227609482404125789454%22%2C%227609060486218403086%22]}; store-country-sign=MEIEDNfhuXsNOHsDdB8LKwQg2g1GBac7EfjjXDYlfGkTdM2xp05RO1AJJBxenZKeluUEEImteJLiPywEp27BDtJDuA4; msToken=hAXqSSx_9Frqxuz3YEuo4NRAl9KXOKaO3G3_yFEymHT6SSfoliIXmSwwQoiR7NDLhrZLqz4lQkdV_W8-BzoTLfsmMKQ6Z80Mo98g9DJklFAtL4hCR09czEJvaR4xU1gJfMNFJ3mDiH09Eo6NmqADzU8V"

# Common request parameters
BASE_PARAMS = {
    "aid": "1988",
    "app_language": "ko-KR",
    "app_name": "tiktok_web",
    "browser_language": "ko-KR",
    "browser_name": "Mozilla",
    "browser_online": "true",
    "browser_platform": "MacIntel",
    "browser_version": "5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    "channel": "tiktok_web",
    "cookie_enabled": "true",
    "device_platform": "web_pc",
    "device_type": "web_h265",
    "os": "mac",
    "priority_region": "US",
    "region": "US",
    "screen_height": "982",
    "screen_width": "1512",
    "tz_name": "America/New_York",
    "webcast_language": "ko-KR",
}

# Request headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://www.tiktok.com/",
    "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "Cookie": COOKIE_STRING,
}
