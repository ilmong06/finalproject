package com.example.wav2vecapp;

import java.util.List;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Query;

public interface ApiService {

    @POST("/api/register_user")
    Call<ResponseBody> registerUser(@Body UserInfo userInfo);

    @GET("/api/user_info")
    Call<UserInfo> getMyInfo();

    @Multipart
    @POST("/api/stt")
    Call<TranscriptionResponse> uploadAudio(@Part MultipartBody.Part file);

    @Multipart
    @POST("/api/register_keyword")
    Call<ResponseBody> registerKeyword(
            @Part("uuid") RequestBody uuid,
            @Part("keyword") RequestBody keyword,
            @Part("order") RequestBody order
    );

    @Multipart
    @POST("/api/register_speaker")
    Call<ResponseBody> registerSpeaker(@Part MultipartBody.Part file);

    // ✅ 추가된 현재 위치 전송 API
    @POST("/api/report_gps")
    Call<ResponseBody> sendGpsData(@Body GpsRequest gpsRequest);

    @GET("/user/check")
    Call<UserResponse> checkUser(@Query("name") String name, @Query("phone") String phone);

    @POST("/api/get_keywords")
    Call<KeywordListResponse> getKeywords(@Body KeywordRequest request);

    @POST("/api/set_selected_keyword")
    Call<ResponseBody> setSelectedKeyword(@Body SelectedKeywordRequest request);
    @Multipart
    @POST("/stt")  // 🔥 Flask 서버에서 정의한 STT 처리 라우트
    Call<ResponseBody> sendSTT(@Part MultipartBody.Part file);
}

