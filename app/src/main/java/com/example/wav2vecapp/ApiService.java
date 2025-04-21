package com.example.wav2vecapp;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface ApiService {
    @Multipart
    @POST("/stt")
    Call<TranscriptionResponse> uploadAudio(@Part MultipartBody.Part file);

    @Multipart
    @POST("/register")
    Call<ResponseBody> registerSpeaker(@Part MultipartBody.Part file);

    @Multipart
    @POST("/register_keyword")
    Call<ResponseBody> registerKeyword(
            @Part MultipartBody.Part file,
            @Part("keyword") RequestBody keyword
    );
}
