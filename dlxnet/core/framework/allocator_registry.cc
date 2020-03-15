#include <string>

#include "dlxnet/core/framework/allocator_registry.h"
#include "dlxnet/core/platform/logging.h"


namespace dlxnet{
    // static
    AllocatorFactoryRegistry* AllocatorFactoryRegistry::singleton() {
        static AllocatorFactoryRegistry* singleton = new AllocatorFactoryRegistry;
        return singleton;
    }

    const AllocatorFactoryRegistry::FactoryEntry*
        AllocatorFactoryRegistry::FindEntry(const string& name, int priority) const {
            for (auto& entry : factories_) {
                if (!name.compare(entry.name) && priority == entry.priority) {
                    return &entry;
                }
            }
            return nullptr;
        }

    void AllocatorFactoryRegistry::Register(const char* source_file,
            int source_line, const string& name,
            int priority,
            AllocatorFactory* factory) {
        mutex_lock l(mu_);
        CHECK(!first_alloc_made_) << "Attempt to register an AllocatorFactory "
            << "after call to GetAllocator()";
        CHECK(!name.empty()) << "Need a valid name for Allocator";
        CHECK_GE(priority, 0) << "Priority needs to be non-negative";

        const FactoryEntry* existing = FindEntry(name, priority);
        if (existing != nullptr) {
            // Duplicate registration is a hard failure.
            LOG(FATAL) << "New registration for AllocatorFactory with name=" << name
                << " priority=" << priority << " at location " << source_file
                << ":" << source_line
                << " conflicts with previous registration at location "
                << existing->source_file << ":" << existing->source_line;
        }

        FactoryEntry entry;
        entry.source_file = source_file;
        entry.source_line = source_line;
        entry.name = name;
        entry.priority = priority;
        entry.factory.reset(factory);
        factories_.push_back(std::move(entry));
    }

    Allocator* AllocatorFactoryRegistry::GetAllocator() {
        mutex_lock l(mu_);
        first_alloc_made_ = true;
        FactoryEntry* best_entry = nullptr;
        for (auto& entry : factories_) {
            if (best_entry == nullptr) {
                best_entry = &entry;
            } else if (entry.priority > best_entry->priority) {
                best_entry = &entry;
            }
        }
        if (best_entry) {
            if (!best_entry->allocator) {
                best_entry->allocator.reset(best_entry->factory->CreateAllocator());
            }
            return best_entry->allocator.get();
        } else {
            LOG(FATAL) << "No registered CPU AllocatorFactory";
            return nullptr;
        }
    }
}
