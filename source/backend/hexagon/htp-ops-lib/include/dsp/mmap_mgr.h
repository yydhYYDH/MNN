#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MmapManager MmapManager;

void *mmap_manager_get_map(int fd);
void *mmap_manager_get_map_local(MmapManager *manager, int fd);
void  mmap_manager_release_all();

MmapManager *mmap_manager_init_local();
void         mmap_manager_destroy_local(MmapManager *manager);
int          mmap_manager_register_local(MmapManager *manager, int fd, void *ptr);

#ifdef __cplusplus
}
#endif
