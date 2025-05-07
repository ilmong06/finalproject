package com.example.wav2vecapp;

import retrofit2.Call;
import retrofit2.http.POST;
import retrofit2.http.Body;
import okhttp3.ResponseBody;

public interface JsonApiService {
    @POST("/api/userinfo")
    Call<ResponseBody> sendUserInfo(@Body UserInfo userInfo);
}

