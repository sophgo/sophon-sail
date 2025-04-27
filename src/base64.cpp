#include "base64.h"

namespace sail {
    //
    // if success, return 0, else return -1
    //
    int base64_enc(Handle& handle, const void *data, uint32_t dlen, std::string& encoded) {
#if USE_BMCV
        if (data == nullptr) return -1;
        bm_handle_t bmHandle = (bm_handle_t) handle.data();
        struct bm_misc_info misc_info;
        bm_status_t ret = bm_get_misc_info(bmHandle, &misc_info);
        if (ret != 0){
            SPDLOG_ERROR("base64_enc err={}, bm_get_misc_info failed", ret);
            throw SailDeviceError("bmlib api fail");
        }
        if (misc_info.pcie_soc_mode == 0 && (misc_info.chipid == 0x1686a200)) {
            SPDLOG_INFO("BM1688 and CV186AH pcie mode not supported");
            throw NotSupport("not supported");
        }
        unsigned long lens[2];
        lens[0] = dlen;
        int out_size = (dlen + 2) / 3 * 4;
        encoded.resize(out_size);
        ret = bmcv_base64_enc(bmHandle, bm_mem_from_system((void *)data),
                              bm_mem_from_system((char *)encoded.data()), lens);
        if (BM_SUCCESS != ret) {
            return -1;
        }

        return 0;
#else
        return -1;
#endif
    }

    //
    // if success, return 0, else return -1
    //
    int base64_dec(Handle& handle, const void *data, uint32_t dlen, uint8_t* p_outbuf, uint32_t *p_size)
    {
#if USE_BMCV
        bm_handle_t bmHandle = (bm_handle_t)handle.data();
        struct bm_misc_info misc_info;
        bm_status_t ret = bm_get_misc_info(bmHandle, &misc_info);
        if (ret != 0){
            SPDLOG_ERROR("base64_enc err={}, bm_get_misc_info failed", ret);
            throw SailDeviceError("bmlib api fail");
        }
        if (misc_info.pcie_soc_mode == 0 && (misc_info.chipid == 0x1686a200)) {
            SPDLOG_INFO("BM1688 and CV186AH pcie mode not supported");
            throw NotSupport("not supported");
        }
        unsigned long lens[2];
        lens[0] = dlen;
        int out_size = dlen/4*3;
        if (nullptr == p_outbuf && nullptr != p_size) {
            *p_size = out_size;
            return -1;
        }

        if (nullptr == data || nullptr == p_size) return -1;
        *p_size = out_size;

        ret = bmcv_base64_dec(bmHandle, bm_mem_from_system((void*)data), bm_mem_from_system(p_outbuf), lens);
        if (BM_SUCCESS != ret) {
            return -1;
        }

        *p_size = lens[1];
        return 0;
#else
        return -1;
#endif
    }
}